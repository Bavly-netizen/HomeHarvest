"""
homeharvest.realtor.__init__
~~~~~~~~~~~~

This module implements the scraper for realtor.com
"""

from __future__ import annotations

import json
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Dict, Union

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    wait_exponential,
    stop_after_attempt,
)

from .. import Scraper, DEFAULT_HEADERS, REALTOR_MAX_RESULTS, SearchMetadata, SearchResult
from ....exceptions import AuthenticationError
from ..models import (
    Property,
    ListingType,
    ReturnType
)
from .queries import GENERAL_RESULTS_QUERY, HOMES_DATA, SEARCH_SUGGESTIONS_QUERY
from .processors import (
    process_property,
    process_extra_property_details,
    get_key
)


class RealtorScraper(Scraper):
    SEARCH_GQL_URL = "https://www.realtor.com/frontdoor/graphql"
    NUM_PROPERTY_WORKERS = 20
    SEARCH_PAGE_WORKERS = 8
    DEFAULT_PAGE_SIZE = 200
    ENRICHMENT_BATCH_SIZE = 20

    def __init__(self, scraper_input):
        super().__init__(scraper_input)

    @staticmethod
    def _minify_query(query: str) -> str:
        """Minify GraphQL query by collapsing whitespace to single spaces."""
        # Split on whitespace, filter empty strings, join with single space
        return ' '.join(query.split())

    def _graphql_post(self, query: str, variables: dict, operation_name: str) -> dict:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string (must include operationName matching operation_name param)
            variables: Query variables dictionary
            operation_name: Name of the GraphQL operation

        Returns:
            Response JSON dictionary
        """
        payload = {
            "operationName": operation_name,
            "query": self._minify_query(query),
            "variables": variables,
        }

        response = requests.post(
            self.SEARCH_GQL_URL,
            headers=DEFAULT_HEADERS,
            data=json.dumps(payload, separators=(',', ':')),
            proxies=self.proxies
        )

        if response.status_code == 403:
            if not self.proxy:
                raise AuthenticationError(
                    "Received 403 Forbidden from Realtor.com API.",
                    response=response
                )
            else:
                raise Exception("Received 403 Forbidden, retrying...")

        return response.json()

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        stop=stop_after_attempt(3),
    )
    def handle_location(self):
        variables = {
            "searchInput": {
                "search_term": self.location
            }
        }

        response_json = self._graphql_post(SEARCH_SUGGESTIONS_QUERY, variables, "Search_suggestions")

        if (
            response_json is None
            or "data" not in response_json
            or response_json["data"] is None
            or "search_suggestions" not in response_json["data"]
            or response_json["data"]["search_suggestions"] is None
            or "geo_results" not in response_json["data"]["search_suggestions"]
            or not response_json["data"]["search_suggestions"]["geo_results"]
        ):
            # If we got a 400 error with "Required parameter is missing", raise to trigger retry
            if response_json and "errors" in response_json:
                error_msgs = [e.get("message", "") for e in response_json.get("errors", [])]
                if any("Required parameter is missing" in msg for msg in error_msgs):
                    raise Exception(f"Transient API error: {error_msgs}")
            return None

        geo_results = response_json["data"]["search_suggestions"]["geo_results"]
        requested_postal_match = re.fullmatch(
            r"(\d{5})(?:-\d{4})?",
            str(self.location or "").strip(),
        )
        if requested_postal_match:
            requested_postal = requested_postal_match.group(1)
            geo_result = next(
                (
                    item
                    for item in geo_results
                    if str((item.get("geo") or {}).get("area_type") or "") == "postal_code"
                    and str((item.get("geo") or {}).get("postal_code") or "")[:5]
                    == requested_postal
                ),
                None,
            )
            if geo_result is None:
                return None
        else:
            geo_result = geo_results[0]
        geo = geo_result.get("geo", {})

        result = {
            "text": geo_result.get("text"),
            "area_type": geo.get("area_type"),
            "city": geo.get("city"),
            "state_code": geo.get("state_code"),
            "postal_code": geo.get("postal_code"),
            "county": geo.get("county"),
            "centroid": geo.get("centroid"),
        }

        if geo.get("area_type") == "address":
            # Try to get mpr_id directly from API response first
            if geo.get("mpr_id"):
                result["mpr_id"] = geo.get("mpr_id")
            else:
                # Fallback: extract from _id field if it has addr: prefix
                geo_id = geo.get("_id", "")
                if geo_id.startswith("addr:"):
                    result["mpr_id"] = geo_id.replace("addr:", "")

        return result

    def get_latest_listing_id(self, property_id: str) -> str | None:
        query = """
                fragment ListingFragment on Listing {
                    listing_id
                    primary
                }
                query GetPropertyListingId($property_id: ID!) {
                    property(id: $property_id) {
                        listings {
                            ...ListingFragment
                        }
                    }
                }
                """

        variables = {"property_id": property_id}
        response_json = self._graphql_post(query, variables, "GetPropertyListingId")

        property_info = response_json["data"]["property"]
        if property_info["listings"] is None:
            return None

        primary_listing = next(
            (listing for listing in property_info["listings"] if listing["primary"]),
            None,
        )
        if primary_listing:
            return primary_listing["listing_id"]
        else:
            return property_info["listings"][0]["listing_id"]

    def handle_home(self, property_id: str) -> list[Property]:
        """Fetch single home with proper error handling."""
        query = (
            """query GetHomeDetails($property_id: ID!) {
                    home(property_id: $property_id) %s
                }"""
            % HOMES_DATA
        )

        variables = {"property_id": property_id}

        try:
            data = self._graphql_post(query, variables, "GetHomeDetails")

            # Check for errors or missing data
            if "errors" in data or "data" not in data:
                return []

            if data["data"] is None or "home" not in data["data"]:
                return []

            property_info = data["data"]["home"]
            if property_info is None:
                return []

            # Process based on return type
            if self.return_type != ReturnType.raw:
                return [process_property(property_info, self.mls_only, self.extra_property_data,
                                       self.exclude_pending, self.listing_type, get_key,
                                       process_extra_property_details)]
            else:
                return [property_info]

        except Exception:
            return []

    def general_search(self, variables: dict, search_type: str) -> Dict[str, Union[int, Union[list[Property], list[dict]]]]:
        """
        Handles a location area & returns a list of properties
        """

        date_param = ""

        # Determine date field based on listing type
        # Convert listing_type to list for uniform handling
        if self.listing_type is None:
            # When None, return all common listing types as documented
            # Note: NEW_COMMUNITY, OTHER, and READY_TO_BUILD are excluded as they typically return no results
            listing_types = [
                ListingType.FOR_SALE,
                ListingType.FOR_RENT,
                ListingType.SOLD,
                ListingType.PENDING,
                ListingType.OFF_MARKET,
            ]
            date_field = None  # When no listing_type is specified, skip date filtering
        elif isinstance(self.listing_type, list):
            listing_types = self.listing_type
            # For multiple types, we'll use a general date field or skip
            date_field = None  # Skip date filtering for mixed types
        else:
            listing_types = [self.listing_type]
            # Determine date field for single type
            if self.listing_type == ListingType.SOLD:
                date_field = "sold_date"
            elif self.listing_type in [ListingType.FOR_SALE, ListingType.FOR_RENT]:
                date_field = "list_date"
            else:  # PENDING or other types
                # Skip server-side date filtering for PENDING as both pending_date and contract_date
                # filters are broken in the API. Client-side filtering will be applied later.
                date_field = None

        # Build date parameter (expand to full days if hour-based filtering is used)
        if date_field:
            # Check if we have hour precision (need to extract date part for API, then filter client-side)
            has_hour_precision = (self.date_from_precision == "hour" or self.date_to_precision == "hour")

            if has_hour_precision and (self.date_from or self.date_to):
                # Hour-based datetime filtering: extract date parts for API, client-side filter by hours
                from datetime import datetime

                min_date = None
                max_date = None

                if self.date_from:
                    try:
                        dt_from = datetime.fromisoformat(self.date_from.replace('Z', '+00:00'))
                        min_date = dt_from.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        pass

                if self.date_to:
                    try:
                        dt_to = datetime.fromisoformat(self.date_to.replace('Z', '+00:00'))
                        max_date = dt_to.strftime("%Y-%m-%d")
                    except (ValueError, AttributeError):
                        pass

                if min_date and max_date:
                    date_param = f'{date_field}: {{ min: "{min_date}", max: "{max_date}" }}'
                elif min_date:
                    date_param = f'{date_field}: {{ min: "{min_date}" }}'
                elif max_date:
                    date_param = f'{date_field}: {{ max: "{max_date}" }}'

            elif self.past_hours:
                # Query API for past N days (minimum 1 day), client-side filter by hours
                days = max(1, int(self.past_hours / 24) + 1)  # Round up to cover the full period
                date_param = f'{date_field}: {{ min: "$today-{days}D" }}'

            elif self.date_from and self.date_to:
                date_param = f'{date_field}: {{ min: "{self.date_from}", max: "{self.date_to}" }}'
            elif self.last_x_days:
                date_param = f'{date_field}: {{ min: "$today-{self.last_x_days}D" }}'

        property_type_param = ""
        if self.property_type:
            property_types = [pt.value for pt in self.property_type]
            property_type_param = f"type: {json.dumps(property_types)}"

        # Build property filter parameters
        property_filters = []

        if self.beds_min is not None or self.beds_max is not None:
            beds_filter = "beds: {"
            if self.beds_min is not None:
                beds_filter += f" min: {self.beds_min}"
            if self.beds_max is not None:
                beds_filter += f" max: {self.beds_max}"
            beds_filter += " }"
            property_filters.append(beds_filter)

        if self.baths_min is not None or self.baths_max is not None:
            baths_filter = "baths: {"
            if self.baths_min is not None:
                baths_filter += f" min: {self.baths_min}"
            if self.baths_max is not None:
                baths_filter += f" max: {self.baths_max}"
            baths_filter += " }"
            property_filters.append(baths_filter)

        if self.sqft_min is not None or self.sqft_max is not None:
            sqft_filter = "sqft: {"
            if self.sqft_min is not None:
                sqft_filter += f" min: {self.sqft_min}"
            if self.sqft_max is not None:
                sqft_filter += f" max: {self.sqft_max}"
            sqft_filter += " }"
            property_filters.append(sqft_filter)

        if self.price_min is not None or self.price_max is not None:
            price_filter = "list_price: {"
            if self.price_min is not None:
                price_filter += f" min: {self.price_min}"
            if self.price_max is not None:
                price_filter += f" max: {self.price_max}"
            price_filter += " }"
            property_filters.append(price_filter)

        if self.lot_sqft_min is not None or self.lot_sqft_max is not None:
            lot_sqft_filter = "lot_sqft: {"
            if self.lot_sqft_min is not None:
                lot_sqft_filter += f" min: {self.lot_sqft_min}"
            if self.lot_sqft_max is not None:
                lot_sqft_filter += f" max: {self.lot_sqft_max}"
            lot_sqft_filter += " }"
            property_filters.append(lot_sqft_filter)

        if self.year_built_min is not None or self.year_built_max is not None:
            year_built_filter = "year_built: {"
            if self.year_built_min is not None:
                year_built_filter += f" min: {self.year_built_min}"
            if self.year_built_max is not None:
                year_built_filter += f" max: {self.year_built_max}"
            year_built_filter += " }"
            property_filters.append(year_built_filter)

        property_filters_param = "\n".join(property_filters)

        # Build sort parameter
        if self.sort_by:
            sort_param = f"sort: [{{ field: {self.sort_by}, direction: {self.sort_direction} }}]"
        elif isinstance(self.listing_type, ListingType) and self.listing_type == ListingType.SOLD:
            sort_param = "sort: [{ field: sold_date, direction: desc }]"
        else:
            sort_param = ""  #: prioritize normal fractal sort from realtor

        # Handle PENDING with or_filters
        # Only use or_filters when PENDING is the only type or mixed only with FOR_SALE
        # Using or_filters with other types (SOLD, FOR_RENT, etc.) will exclude those types
        has_pending = ListingType.PENDING in listing_types
        other_types = [lt for lt in listing_types if lt not in [ListingType.PENDING, ListingType.FOR_SALE]]
        use_or_filters = has_pending and len(other_types) == 0
        pending_or_contingent_param = (
            "or_filters: { contingent: true, pending: true }" if use_or_filters else ""
        )

        # Build bucket parameter (only use fractal sort if no custom sort is specified)
        bucket_param = ""
        if not self.sort_by:
            bucket_param = 'bucket: { sort: "fractal_v1.1.3_fr" }'

        # Build status parameter
        # For PENDING, we need to query as FOR_SALE with or_filters for pending/contingent
        status_types = []
        for lt in listing_types:
            if lt == ListingType.PENDING:
                if ListingType.FOR_SALE not in status_types:
                    status_types.append(ListingType.FOR_SALE)
            else:
                if lt not in status_types:
                    status_types.append(lt)

        # Build status parameter string
        if status_types:
            status_values = [st.value.lower() for st in status_types]
            if len(status_values) == 1:
                status_param = f"status: {status_values[0]}"
            else:
                status_param = f"status: [{', '.join(status_values)}]"
        else:
            status_param = ""  # No status parameter means return all types

        is_foreclosure = ""

        if variables.get("foreclosure") is True:
            is_foreclosure = "foreclosure: true"
        elif variables.get("foreclosure") is False:
            is_foreclosure = "foreclosure: false"

        if search_type == "comps":  #: comps search, came from an address
            query = """query GetHomeSearch(
                    $coordinates: [Float]!
                    $radius: String!
                    $offset: Int!,
                    ) {
                        homeSearch: home_search(
                            query: {
                                %s
                                nearby: {
                                    coordinates: $coordinates
                                    radius: $radius
                                }
                                %s
                                %s
                                %s
                                %s
                                %s
                            }
                            %s
                            limit: 200
                            offset: $offset
                    ) %s
                }""" % (
                is_foreclosure,
                status_param,
                date_param,
                property_type_param,
                property_filters_param,
                pending_or_contingent_param,
                sort_param,
                GENERAL_RESULTS_QUERY,
            )
        elif search_type == "area":  #: general search, came from a general location
            query = """query GetHomeSearch(
                                $search_location: SearchLocation,
                                $postal_code: String,
                                $offset: Int
                            ) {
                                homeSearch: home_search(
                                    query: {
                                        %s
                                        search_location: $search_location
                                        postal_code: $postal_code
                                        %s
                                        %s
                                        %s
                                        %s
                                        %s
                                    }
                                    %s
                                    %s
                                    limit: 200
                                    offset: $offset
                                ) %s
                            }""" % (
                is_foreclosure,
                status_param,
                date_param,
                property_type_param,
                property_filters_param,
                pending_or_contingent_param,
                bucket_param,
                sort_param,
                GENERAL_RESULTS_QUERY,
            )
        else:  #: general search, came from an address
            query = (
                """query GetHomeSearch(
                        $property_id: [ID]!
                        $offset: Int!,
                    ) {
                        homeSearch: home_search(
                            query: {
                                property_id: $property_id
                            }
                            limit: 1
                            offset: $offset
                        ) %s
                    }"""
                % GENERAL_RESULTS_QUERY
            )

        response_json = self._graphql_post(query, variables, "GetHomeSearch")
        search_key = "homeSearch"

        properties: list[Union[Property, dict]] = []

        page_offset = variables.get("offset", 0)
        window_end = self.offset + self.limit
        page_window_size = max(0, min(self.DEFAULT_PAGE_SIZE, window_end - page_offset))

        if (
            response_json is None
            or "data" not in response_json
            or response_json["data"] is None
            or search_key not in response_json["data"]
            or response_json["data"][search_key] is None
            or "results" not in response_json["data"][search_key]
        ):
            # Never echo arbitrary GraphQL/server error text into metadata; the
            # caller gets a fixed, safe category so proxy URLs or credentials
            # cannot leak back through SearchMetadata.errors.
            return {
                "total": 0,
                "properties": [],
                "raw_rows": 0,
                "window_rows": 0,
                "processed_rows": 0,
                "processor_rejected_rows": 0,
                "offset": page_offset,
                "requested_limit": page_window_size,
                "completed": False,
                "error": "GraphQL error: page request failed",
                "enrichment_requested_ids": 0,
                "enrichment_received_ids": 0,
                "enrichment_missing_ids": 0,
                "enrichment_unaddressable_rows": 0,
                "enrichment_errors": [],
            }

        properties_list = response_json["data"][search_key]["results"]
        total_properties = response_json["data"][search_key]["total"]

        #: raw_rows: source rows received before client-side truncation / processing
        raw_rows = len(properties_list)

        #: Keep only the portion of the page that falls inside the requested window.
        #: The API always returns a page starting at page_offset; we consume up to
        #: the page size or the remaining window, whichever is smaller.
        properties_list: list[dict] = properties_list[:page_window_size]

        if self.extra_property_data:
            property_ids: list[str] = []
            seen_ids: set[str] = set()
            enrichment_unaddressable_rows = 0
            for data in properties_list:
                property_id = data.get("property_id")
                if property_id is None or (isinstance(property_id, str) and not property_id.strip()):
                    enrichment_unaddressable_rows += 1
                elif property_id not in seen_ids:
                    seen_ids.add(property_id)
                    property_ids.append(property_id)

            enrichment_requested_ids = len(property_ids)
            enrichment_received_ids = 0
            enrichment_missing_ids = 0
            enrichment_errors: list[str] = []
            extra_property_details: dict = {}

            if enrichment_unaddressable_rows > 0:
                enrichment_errors.append("enrichment: rows missing property id")

            batch_had_exception = False
            for batch_start in range(0, len(property_ids), self.ENRICHMENT_BATCH_SIZE):
                batch = property_ids[batch_start:batch_start + self.ENRICHMENT_BATCH_SIZE]
                if not batch:
                    continue
                try:
                    batch_details = self.get_bulk_prop_details(batch) or {}
                except Exception:
                    batch_had_exception = True
                    if "enrichment: bulk details request failed" not in enrichment_errors:
                        enrichment_errors.append("enrichment: bulk details request failed")
                    continue
                extra_property_details.update(batch_details)

            enrichment_received_ids = sum(
                1 for property_id in property_ids
                if extra_property_details.get(property_id)
            )
            enrichment_missing_ids = enrichment_requested_ids - enrichment_received_ids
            if enrichment_requested_ids > 0 and enrichment_received_ids == 0 and not batch_had_exception:
                if "enrichment: empty details response" not in enrichment_errors:
                    enrichment_errors.append("enrichment: empty details response")
            elif enrichment_missing_ids > 0 and enrichment_received_ids > 0:
                if "enrichment: partial details response" not in enrichment_errors:
                    enrichment_errors.append("enrichment: partial details response")

            for result in properties_list:
                property_id = result.get("property_id")
                if not property_id or (isinstance(property_id, str) and not property_id.strip()):
                    continue
                specific_details_for_property = extra_property_details.get(property_id, {})
                if not specific_details_for_property:
                    continue

                #: address is retrieved on both homes and search homes, so when merged, homes overrides,
                # this gets the internal data we want and only updates that (migrate to a func if more fields)
                if "location" in specific_details_for_property:
                    result["location"].update(specific_details_for_property["location"])
                    del specific_details_for_property["location"]

                result.update(specific_details_for_property)
        else:
            enrichment_requested_ids = 0
            enrichment_received_ids = 0
            enrichment_missing_ids = 0
            enrichment_unaddressable_rows = 0
            enrichment_errors = []

        if self.return_type != ReturnType.raw:
            with ThreadPoolExecutor(max_workers=self.NUM_PROPERTY_WORKERS) as executor:
                # Store futures with their indices to maintain sort order
                futures_with_indices = [
                    (i, executor.submit(process_property, result, self.mls_only, self.extra_property_data,
                                       self.exclude_pending, self.listing_type, get_key, process_extra_property_details))
                    for i, result in enumerate(properties_list)
                ]

                # Collect results and sort by index to preserve API sort order
                results = []
                for idx, future in futures_with_indices:
                    result = future.result()
                    if result:
                        results.append((idx, result))

                # Sort by index and extract properties in correct order
                results.sort(key=lambda x: x[0])
                properties = [result for idx, result in results]
                processed_rows = len(properties)
                processor_rejected_rows = len(properties_list) - processed_rows
        else:
            properties = properties_list
            processed_rows = len(properties_list)
            processor_rejected_rows = 0

        return {
            "total": total_properties,
            "properties": properties,
            "raw_rows": raw_rows,
            "window_rows": len(properties_list),
            "processed_rows": processed_rows,
            "processor_rejected_rows": processor_rejected_rows,
            "offset": page_offset,
            "requested_limit": page_window_size,
            "completed": True,
            "error": None,
            "enrichment_requested_ids": enrichment_requested_ids,
            "enrichment_received_ids": enrichment_received_ids,
            "enrichment_missing_ids": enrichment_missing_ids,
            "enrichment_unaddressable_rows": enrichment_unaddressable_rows,
            "enrichment_errors": enrichment_errors,
        }

    def search(self):
        location_info = self.handle_location()
        if not location_info:
            metadata = SearchMetadata(
                requested_limit=self.limit,
                requested_offset=self.offset,
                completeness_proven=False,
                errors=["Location could not be resolved"],
            )
            return self._search_result([], metadata)

        location_type = location_info["area_type"]

        search_variables = {
            "offset": self.offset,
        }

        search_type = (
            "comps"
            if self.radius and location_type == "address"
            else "address" if location_type == "address" and not self.radius else "area"
        )
        if location_type == "address":
            if not self.radius:  #: single address search, non comps
                property_id = location_info["mpr_id"]
                homes = self.handle_home(property_id)
                total = 1 if homes else 0
                raw_rows = len(homes)
                metadata = SearchMetadata(
                    source_reported_total=total,
                    requested_limit=self.limit,
                    requested_offset=self.offset,
                    raw_rows_received=raw_rows,
                    window_rows_received=raw_rows,
                    processed_rows_received=raw_rows,
                    processor_rejected_rows=0,
                    returned_rows=raw_rows,
                    page_offsets_attempted=[0],
                    page_offsets_completed=[0] if homes else [],
                    page_offsets_failed=[] if homes else [0],
                    base_completeness_proven=bool(homes) and len(homes) == 1,
                    full_base_result_set_completeness_proven=bool(homes) and len(homes) == 1,
                    enrichment_requested=False,
                    enrichment_completeness_proven=True,
                    completeness_proven=bool(homes) and len(homes) == 1,
                    full_result_set_completeness_proven=bool(homes) and len(homes) == 1,
                    errors=[] if homes else ["Single address search returned no home"],
                )
                return self._search_result(homes, metadata)

            else:  #: general search, comps (radius)
                if not location_info.get("centroid"):
                    metadata = SearchMetadata(
                        requested_limit=self.limit,
                        requested_offset=self.offset,
                        completeness_proven=False,
                        errors=["No centroid for address radius search"],
                    )
                    return self._search_result([], metadata)

                centroid = location_info["centroid"]
                coordinates = [centroid["lon"], centroid["lat"]]  # GeoJSON order: [lon, lat]
                search_variables |= {
                    "coordinates": coordinates,
                    "radius": "{}mi".format(self.radius),
                }

        else:  #: general search (city, county, postal_code, etc.)
            if location_type == "postal_code":
                postal_code = str(location_info.get("postal_code") or "").strip()[:5]
                requested_postal_match = re.fullmatch(
                    r"(\d{5})(?:-\d{4})?",
                    str(self.location or "").strip(),
                )
                if (
                    not requested_postal_match
                    or postal_code != requested_postal_match.group(1)
                ):
                    metadata = SearchMetadata(
                        requested_limit=self.limit,
                        requested_offset=self.offset,
                        completeness_proven=False,
                        errors=["Postal location did not resolve exactly"],
                    )
                    return self._search_result([], metadata)
                search_variables["postal_code"] = postal_code
            else:
                search_variables["search_location"] = {
                    "location": location_info.get("text")
                }

        if self.foreclosure:
            search_variables["foreclosure"] = self.foreclosure

        try:
            result = self.general_search(search_variables, search_type=search_type)
        except Exception as exc:
            result = self._failed_page(self.offset, exc)
        page_results = [result]
        total = result["total"]
        homes = result["properties"]

        # Fetch remaining pages based on parallel parameter
        if self.offset + self.DEFAULT_PAGE_SIZE < min(total, self.offset + self.limit):
            if self.parallel:
                # Parallel mode: Fetch all remaining pages in parallel
                with ThreadPoolExecutor(max_workers=self.SEARCH_PAGE_WORKERS) as executor:
                    futures_with_offsets = [
                        (i, executor.submit(
                            self.general_search,
                            variables=search_variables | {"offset": i},
                            search_type=search_type,
                        ))
                        for i in range(
                            self.offset + self.DEFAULT_PAGE_SIZE,
                            min(total, self.offset + self.limit),
                            self.DEFAULT_PAGE_SIZE,
                        )
                    ]

                    # Collect results, capture exceptions, and sort by offset to preserve order
                    results = []
                    for offset, future in futures_with_offsets:
                        try:
                            page_result = future.result()
                        except Exception as exc:
                            page_result = self._failed_page(offset, exc)
                        results.append((offset, page_result))

                    results.sort(key=lambda x: x[0])
                    for offset, page_result in results:
                        page_results.append(page_result)
                        homes.extend(page_result["properties"])
            else:
                # Sequential mode: Fetch pages one by one with early termination checks
                for current_offset in range(
                    self.offset + self.DEFAULT_PAGE_SIZE,
                    min(total, self.offset + self.limit),
                    self.DEFAULT_PAGE_SIZE,
                ):
                    # Check if we should continue based on time-based filters
                    if not self._should_fetch_more_pages(homes):
                        break

                    try:
                        page_result = self.general_search(
                            variables=search_variables | {"offset": current_offset},
                            search_type=search_type,
                        )
                    except Exception as exc:
                        page_result = self._failed_page(current_offset, exc)
                    page_results.append(page_result)
                    homes.extend(page_result["properties"])

        # Apply client-side hour-based filtering if needed
        # (API only supports day-level filtering, so we post-filter for hour precision)
        has_hour_precision = (self.date_from_precision == "hour" or self.date_to_precision == "hour")
        if self.past_hours or has_hour_precision:
            homes = self._apply_hour_based_date_filter(homes)
        # Apply client-side date filtering for PENDING properties
        # (server-side filters are broken in the API)
        elif self.listing_type == ListingType.PENDING and (self.last_x_days or self.date_from):
            homes = self._apply_pending_date_filter(homes)

        # Apply client-side filtering by last_update_date if specified
        if self.updated_since or self.updated_in_past_hours:
            homes = self._apply_last_update_date_filter(homes)

        # Apply client-side sort to ensure results are properly ordered
        # This is necessary after filtering and to guarantee sort order across page boundaries
        if self.sort_by:
            homes = self._apply_sort(homes)

        # Apply raw data filters (exclude_pending and mls_only) for raw return type
        # These filters are normally applied in process_property() but are bypassed for raw data
        if self.return_type == ReturnType.raw:
            homes = self._apply_raw_data_filters(homes)

        metadata = self._build_search_metadata(page_results, len(homes))
        return self._search_result(homes, metadata)

    def _failed_page(self, offset: int, error: Exception) -> dict:
        """Build a failed page result so exceptions become deterministic metadata."""
        page_window_size = max(0, min(self.DEFAULT_PAGE_SIZE, self.offset + self.limit - offset))
        return {
            "total": 0,
            "properties": [],
            "raw_rows": 0,
            "window_rows": 0,
            "processed_rows": 0,
            "processor_rejected_rows": 0,
            "offset": offset,
            "requested_limit": page_window_size,
            "completed": False,
            "error": f"{type(error).__name__}: page request failed",
            "enrichment_requested_ids": 0,
            "enrichment_received_ids": 0,
            "enrichment_missing_ids": 0,
            "enrichment_unaddressable_rows": 0,
            "enrichment_errors": [],
        }

    def _build_search_metadata(self, page_results: list[dict], returned_rows: int) -> SearchMetadata:
        """Aggregate per-page results into a single SearchMetadata object."""
        attempted: list[int] = []
        completed: list[int] = []
        failed: list[int] = []
        raw_rows_received = 0
        window_rows_received = 0
        processed_rows_received = 0
        processor_rejected_rows = 0
        errors: list[str] = []
        enrichment_requested_ids = 0
        enrichment_received_ids = 0
        enrichment_missing_ids = 0
        enrichment_unaddressable_rows = 0
        enrichment_errors: list[str] = []
        source_reported_total: int | None = None
        observed_totals: list[tuple[int, int]] = []
        enrichment_requested = bool(self.extra_property_data)

        for page in page_results:
            offset = page.get("offset", 0)
            attempted.append(offset)
            if page.get("completed"):
                completed.append(offset)
                raw_rows_received += page.get("raw_rows", 0)
                window_rows_received += page.get("window_rows", 0)
                processed_rows_received += page.get("processed_rows", 0)
                processor_rejected_rows += page.get("processor_rejected_rows", 0)
                enrichment_requested_ids += page.get("enrichment_requested_ids", 0)
                enrichment_received_ids += page.get("enrichment_received_ids", 0)
                enrichment_missing_ids += page.get("enrichment_missing_ids", 0)
                enrichment_unaddressable_rows += page.get("enrichment_unaddressable_rows", 0)
                for err in page.get("enrichment_errors", []):
                    if err not in enrichment_errors:
                        enrichment_errors.append(err)
                page_total = page.get("total")
                if page_total is not None:
                    observed_totals.append((offset, page_total))
                    if source_reported_total is None:
                        source_reported_total = page_total
            else:
                failed.append(offset)
                err = page.get("error")
                if err:
                    errors.append(f"offset {offset}: {err}")

        if observed_totals:
            first_total = observed_totals[0][1]
            for offset, page_total in observed_totals[1:]:
                if page_total != first_total:
                    errors.append(
                        f"offset {offset}: inconsistent source total {page_total} "
                        f"(first page reported {first_total})"
                    )

        requested_offset = self.offset
        requested_limit = self.limit

        # Cap/truncation decisions must be based on window rows (the raw source
        # rows inside the caller's requested window), not on transport overfetch.
        # A 9,900-row window whose source total is 9,900 may receive 10,000
        # transport rows; that is complete, not capped.
        reached_10k_boundary = (
            requested_offset + requested_limit >= REALTOR_MAX_RESULTS
            or (source_reported_total is not None and source_reported_total >= REALTOR_MAX_RESULTS)
            or window_rows_received >= REALTOR_MAX_RESULTS
            or any(o >= REALTOR_MAX_RESULTS for o in attempted)
        )

        truncated_by_10k = (
            window_rows_received >= REALTOR_MAX_RESULTS
            or (source_reported_total is not None and source_reported_total >= REALTOR_MAX_RESULTS)
            or (
                requested_offset + requested_limit >= REALTOR_MAX_RESULTS
                and (source_reported_total is None or source_reported_total > requested_offset + requested_limit)
            )
        )

        expected_rows: int | None = None
        expected_offsets: list[int] = []
        if source_reported_total is not None:
            expected_rows = max(0, min(source_reported_total - requested_offset, requested_limit))
            expected_offsets = [
                o
                for o in range(requested_offset, requested_offset + requested_limit, self.DEFAULT_PAGE_SIZE)
                if o < source_reported_total
            ]

        all_expected_offsets_attempted = all(o in attempted for o in expected_offsets)

        # Base listing-set completeness ignores optional enrichment failures.
        base_completeness_proven = (
            not truncated_by_10k
            and not errors
            and not failed
            and expected_rows is not None
            and window_rows_received == expected_rows
            and all_expected_offsets_attempted
            and processor_rejected_rows == 0
        )
        full_base_result_set_completeness_proven = (
            base_completeness_proven
            and requested_offset == 0
            and source_reported_total is not None
            and requested_limit >= source_reported_total
            and returned_rows == source_reported_total
        )

        if enrichment_requested:
            enrichment_completeness_proven = (
                enrichment_unaddressable_rows == 0
                and enrichment_missing_ids == 0
                and not enrichment_errors
                and enrichment_received_ids == enrichment_requested_ids
            )
        else:
            enrichment_completeness_proven = True

        completeness_proven = base_completeness_proven and enrichment_completeness_proven
        full_result_set_completeness_proven = (
            full_base_result_set_completeness_proven and enrichment_completeness_proven
        )

        return SearchMetadata(
            source_reported_total=source_reported_total,
            requested_limit=requested_limit,
            requested_offset=requested_offset,
            raw_rows_received=raw_rows_received,
            window_rows_received=window_rows_received,
            processed_rows_received=processed_rows_received,
            processor_rejected_rows=processor_rejected_rows,
            returned_rows=returned_rows,
            page_offsets_attempted=attempted,
            page_offsets_completed=completed,
            page_offsets_failed=failed,
            reached_10k_boundary=reached_10k_boundary,
            truncated_by_10k=truncated_by_10k,
            base_completeness_proven=base_completeness_proven,
            full_base_result_set_completeness_proven=full_base_result_set_completeness_proven,
            enrichment_requested=enrichment_requested,
            enrichment_requested_ids=enrichment_requested_ids,
            enrichment_received_ids=enrichment_received_ids,
            enrichment_missing_ids=enrichment_missing_ids,
            enrichment_unaddressable_rows=enrichment_unaddressable_rows,
            enrichment_completeness_proven=enrichment_completeness_proven,
            enrichment_errors=enrichment_errors,
            completeness_proven=completeness_proven,
            full_result_set_completeness_proven=full_result_set_completeness_proven,
            errors=errors,
        )

    def _search_result(self, properties, metadata):
        """Return either the raw list or a SearchResult wrapper depending on the caller's opt-in."""
        if self.return_metadata:
            return SearchResult(properties=properties, metadata=metadata)
        return properties

    def _apply_hour_based_date_filter(self, homes):
        """Apply client-side hour-based date filtering for all listing types.

        This is used when past_hours or date_from/date_to have hour precision,
        since the API only supports day-level filtering.
        """
        if not homes:
            return homes

        from datetime import datetime, timedelta

        # Determine date range with hour precision
        date_range = None

        if self.past_hours:
            cutoff_datetime = datetime.now() - timedelta(hours=self.past_hours)
            date_range = {'type': 'since', 'date': cutoff_datetime}
        elif self.date_from or self.date_to:
            try:
                from_datetime = None
                to_datetime = None

                if self.date_from:
                    from_datetime_str = self.date_from.replace('Z', '+00:00') if self.date_from.endswith('Z') else self.date_from
                    from_datetime = datetime.fromisoformat(from_datetime_str).replace(tzinfo=None)

                if self.date_to:
                    to_datetime_str = self.date_to.replace('Z', '+00:00') if self.date_to.endswith('Z') else self.date_to
                    to_datetime = datetime.fromisoformat(to_datetime_str).replace(tzinfo=None)

                if from_datetime and to_datetime:
                    date_range = {'type': 'range', 'from_date': from_datetime, 'to_date': to_datetime}
                elif from_datetime:
                    date_range = {'type': 'since', 'date': from_datetime}
                elif to_datetime:
                    date_range = {'type': 'until', 'date': to_datetime}
            except (ValueError, AttributeError):
                return homes  # If parsing fails, return unfiltered

        if not date_range:
            return homes

        # Determine which date field to use based on listing type
        date_field_name = self._get_date_field_for_listing_type()

        filtered_homes = []

        for home in homes:
            # Extract the appropriate date for this property
            property_date = self._extract_date_from_home(home, date_field_name)

            # Handle properties without dates
            if property_date is None:
                # For PENDING, include contingent properties without pending_date
                if self.listing_type == ListingType.PENDING and self._is_contingent(home):
                    filtered_homes.append(home)
                continue

            # Check if property date falls within the specified range
            if self._is_datetime_in_range(property_date, date_range):
                filtered_homes.append(home)

        return filtered_homes

    def _get_date_field_for_listing_type(self):
        """Get the appropriate date field name for the current listing type."""
        if self.listing_type == ListingType.SOLD:
            return 'last_sold_date'
        elif self.listing_type == ListingType.PENDING:
            return 'pending_date'
        else:  # FOR_SALE or FOR_RENT
            return 'list_date'

    def _extract_date_from_home(self, home, date_field_name):
        """Extract a date field from a home (handles both dict and Property object).

        Falls back to last_status_change_date if the primary date field is not available,
        providing more precise filtering for PENDING/SOLD properties.
        """
        if isinstance(home, dict):
            date_value = home.get(date_field_name)
        else:
            date_value = getattr(home, date_field_name, None)

        if date_value:
            return self._parse_date_value(date_value)

        # Fallback to last_status_change_date if primary date field is missing
        # This is useful for PENDING/SOLD properties where the specific date might be unavailable
        if isinstance(home, dict):
            fallback_date = home.get('last_status_change_date')
        else:
            fallback_date = getattr(home, 'last_status_change_date', None)

        if fallback_date:
            return self._parse_date_value(fallback_date)

        return None

    def _is_datetime_in_range(self, date_obj, date_range):
        """Check if a datetime object falls within the specified date range (with hour precision)."""
        if date_range['type'] == 'since':
            return date_obj >= date_range['date']
        elif date_range['type'] == 'until':
            return date_obj <= date_range['date']
        elif date_range['type'] == 'range':
            return date_range['from_date'] <= date_obj <= date_range['to_date']
        return False

    def _apply_pending_date_filter(self, homes):
        """Apply client-side date filtering for PENDING properties based on pending_date field.
        For contingent properties without pending_date, tries fallback date fields."""
        if not homes:
            return homes
            
        from datetime import datetime, timedelta
        
        # Determine date range for filtering
        date_range = self._get_date_range()
        if not date_range:
            return homes
            
        filtered_homes = []
        
        for home in homes:
            # Extract the best available date for this property
            property_date = self._extract_property_date_for_filtering(home)
            
            # Handle properties without dates (include contingent properties)
            if property_date is None:
                if self._is_contingent(home):
                    filtered_homes.append(home)  # Include contingent without date filter
                continue
            
            # Check if property date falls within the specified range
            if self._is_date_in_range(property_date, date_range):
                filtered_homes.append(home)
                
        return filtered_homes
    
    def _get_pending_date(self, home):
        """Extract pending_date from a home property (handles both dict and Property object)."""
        if isinstance(home, dict):
            return home.get('pending_date')
        else:
            # Assume it's a Property object
            return getattr(home, 'pending_date', None)
    
    
    def _is_contingent(self, home):
        """Check if a property is contingent."""
        if isinstance(home, dict):
            flags = home.get('flags', {})
            return flags.get('is_contingent', False)
        else:
            # Property object - check flags attribute
            if hasattr(home, 'flags') and home.flags:
                return getattr(home.flags, 'is_contingent', False)
            return False

    def _apply_last_update_date_filter(self, homes):
        """Apply client-side filtering by last_update_date.

        This is used when updated_since or updated_in_past_hours are specified.
        Filters properties based on when they were last updated.
        """
        if not homes:
            return homes

        from datetime import datetime, timedelta, timezone

        # Determine date range for last_update_date filtering
        date_range = None

        if self.updated_in_past_hours:
            # Use UTC now, strip timezone to match naive property dates
            cutoff_datetime = (datetime.now(timezone.utc) - timedelta(hours=self.updated_in_past_hours)).replace(tzinfo=None)
            date_range = {'type': 'since', 'date': cutoff_datetime}
        elif self.updated_since:
            try:
                since_datetime_str = self.updated_since.replace('Z', '+00:00') if self.updated_since.endswith('Z') else self.updated_since
                since_datetime = datetime.fromisoformat(since_datetime_str).replace(tzinfo=None)
                date_range = {'type': 'since', 'date': since_datetime}
            except (ValueError, AttributeError):
                return homes  # If parsing fails, return unfiltered

        if not date_range:
            return homes

        filtered_homes = []

        for home in homes:
            # Extract last_update_date from the property
            property_date = self._extract_date_from_home(home, 'last_update_date')

            # Skip properties without last_update_date
            if property_date is None:
                continue

            # Check if property date falls within the specified range
            if self._is_datetime_in_range(property_date, date_range):
                filtered_homes.append(home)

        return filtered_homes

    def _get_date_range(self):
        """Get the date range for filtering based on instance parameters."""
        from datetime import datetime, timedelta, timezone

        if self.last_x_days:
            # Use UTC now, strip timezone to match naive property dates
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=self.last_x_days)).replace(tzinfo=None)
            return {'type': 'since', 'date': cutoff_date}
        elif self.date_from and self.date_to:
            try:
                # Parse and strip timezone to match naive property dates
                from_date_str = self.date_from.replace('Z', '+00:00') if self.date_from.endswith('Z') else self.date_from
                to_date_str = self.date_to.replace('Z', '+00:00') if self.date_to.endswith('Z') else self.date_to
                from_date = datetime.fromisoformat(from_date_str).replace(tzinfo=None)
                to_date = datetime.fromisoformat(to_date_str).replace(tzinfo=None)
                return {'type': 'range', 'from_date': from_date, 'to_date': to_date}
            except ValueError:
                return None
        return None
    
    def _extract_property_date_for_filtering(self, home):
        """Extract pending_date from a property for filtering.
        
        Returns parsed datetime object or None.
        """
        date_value = self._get_pending_date(home)
        if date_value:
            return self._parse_date_value(date_value)
        return None
    
    def _parse_date_value(self, date_value):
        """Parse a date value (string or datetime) into a timezone-naive datetime object."""
        from datetime import datetime
        
        if isinstance(date_value, datetime):
            return date_value.replace(tzinfo=None)
        
        if not isinstance(date_value, str):
            return None
            
        try:
            # Handle timezone indicators
            if date_value.endswith('Z'):
                date_value = date_value[:-1] + '+00:00'
            elif '.' in date_value and date_value.endswith('Z'):
                date_value = date_value.replace('Z', '+00:00')
            
            # Try ISO format first
            try:
                parsed_date = datetime.fromisoformat(date_value)
                return parsed_date.replace(tzinfo=None)
            except ValueError:
                # Try simple datetime format: '2025-08-29 00:00:00'
                return datetime.strptime(date_value, '%Y-%m-%d %H:%M:%S')
                
        except (ValueError, AttributeError):
            return None
    
    def _is_date_in_range(self, date_obj, date_range):
        """Check if a datetime object falls within the specified date range."""
        if date_range['type'] == 'since':
            return date_obj >= date_range['date']
        elif date_range['type'] == 'range':
            return date_range['from_date'] <= date_obj <= date_range['to_date']
        return False

    def _should_fetch_more_pages(self, first_page):
        """Determine if we should continue pagination based on first page results.

        This optimization prevents unnecessary API calls when using time-based filters
        with date sorting. If the last property on page 1 is already outside the time
        window, all future pages will also be outside (due to sort order).

        Args:
            first_page: List of properties from the first page

        Returns:
            bool: True if we should continue pagination, False to stop early
        """
        from datetime import datetime, timedelta, timezone

        # Check for last_update_date filters
        if (self.updated_since or self.updated_in_past_hours) and self.sort_by == "last_update_date":
            if not first_page:
                return False

            last_property = first_page[-1]
            last_date = self._extract_date_from_home(last_property, 'last_update_date')

            if not last_date:
                return True

            # Build date range for last_update_date filter
            if self.updated_since:
                try:
                    cutoff_datetime = datetime.fromisoformat(self.updated_since.replace('Z', '+00:00') if self.updated_since.endswith('Z') else self.updated_since)
                    # Strip timezone to match naive datetimes from _parse_date_value
                    cutoff_datetime = cutoff_datetime.replace(tzinfo=None)
                    date_range = {'type': 'since', 'date': cutoff_datetime}
                except ValueError:
                    return True
            elif self.updated_in_past_hours:
                # Use UTC now, strip timezone to match naive property dates
                cutoff_datetime = (datetime.now(timezone.utc) - timedelta(hours=self.updated_in_past_hours)).replace(tzinfo=None)
                date_range = {'type': 'since', 'date': cutoff_datetime}
            else:
                return True

            return self._is_datetime_in_range(last_date, date_range)

        # Check for PENDING date filters
        if (self.listing_type == ListingType.PENDING and
            (self.last_x_days or self.past_hours or self.date_from) and
            self.sort_by == "pending_date"):

            if not first_page:
                return False

            last_property = first_page[-1]
            last_date = self._extract_date_from_home(last_property, 'pending_date')

            if not last_date:
                return True

            # Build date range for pending date filter
            date_range = self._get_date_range()
            if not date_range:
                return True

            return self._is_datetime_in_range(last_date, date_range)

        # No optimization applicable, continue pagination
        return True

    def _apply_sort(self, homes):
        """Apply client-side sorting to ensure results are properly ordered.

        This is necessary because:
        1. Multi-page results need to be re-sorted after concatenation
        2. Filtering operations may disrupt the original sort order

        Args:
            homes: List of properties (either dicts or Property objects)

        Returns:
            Sorted list of properties
        """
        if not homes or not self.sort_by:
            return homes

        def get_sort_key(home):
            """Extract the sort field value from a home (handles both dict and Property object)."""
            from datetime import datetime

            if isinstance(home, dict):
                value = home.get(self.sort_by)
            else:
                # Property object
                value = getattr(home, self.sort_by, None)

            # Handle None values - push them to the end
            if value is None:
                # Use a sentinel value that sorts to the end
                return (1, 0) if self.sort_direction == "desc" else (1, float('inf'))

            # For datetime fields, convert string to datetime for proper sorting
            if self.sort_by in ['list_date', 'sold_date', 'pending_date', 'last_update_date']:
                if isinstance(value, str):
                    try:
                        # Handle timezone indicators
                        date_value = value
                        if date_value.endswith('Z'):
                            date_value = date_value[:-1] + '+00:00'
                        parsed_date = datetime.fromisoformat(date_value)
                        # Normalize to timezone-naive for consistent comparison
                        return 0, parsed_date.replace(tzinfo=None)
                    except (ValueError, AttributeError):
                        # If parsing fails, treat as None
                        return (1, 0) if self.sort_direction == "desc" else (1, float('inf'))
                # Handle datetime objects directly (normalize timezone)
                if isinstance(value, datetime):
                    return 0, value.replace(tzinfo=None)
                return 0, value

            # For numeric fields, ensure we can compare
            return 0, value

        # Sort the homes
        reverse = (self.sort_direction == "desc")
        sorted_homes = sorted(homes, key=get_sort_key, reverse=reverse)

        return sorted_homes

    def _apply_raw_data_filters(self, homes):
        """Apply exclude_pending and mls_only filters for raw data returns.

        These filters are normally applied in process_property(), but that function
        is bypassed when return_type="raw", so we need to apply them here instead.

        Args:
            homes: List of properties (either dicts or Property objects)

        Returns:
            Filtered list of properties
        """
        if not homes:
            return homes

        # Only filter raw data (dict objects)
        # Property objects have already been filtered in process_property()
        if homes and not isinstance(homes[0], dict):
            return homes

        filtered_homes = []

        for home in homes:
            # Apply exclude_pending filter
            if self.exclude_pending and self.listing_type != ListingType.PENDING:
                flags = home.get('flags', {})
                is_pending = flags.get('is_pending', False)
                is_contingent = flags.get('is_contingent', False)

                if is_pending or is_contingent:
                    continue  # Skip this property

            # Apply mls_only filter
            if self.mls_only:
                source = home.get('source', {})
                if not source or not source.get('id'):
                    continue  # Skip this property

            filtered_homes.append(home)

        return filtered_homes


    @retry(
        retry=retry_if_exception_type((JSONDecodeError, Exception)) & retry_if_not_exception_type(AuthenticationError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def get_bulk_prop_details(self, property_ids: list[str]) -> dict:
        """
        Fetch extra property details for multiple properties in a single GraphQL query.
        Returns a map of property_id to its details.
        """
        if not self.extra_property_data or not property_ids:
            return {}

        property_ids = list(set(property_ids))

        fragments = "\n".join(
            f'home_{property_id}: home(property_id: {property_id}) {HOMES_DATA}'
            for property_id in property_ids
        )
        query = f"""query GetHome {{
    {fragments}
}}"""

        data = self._graphql_post(query, {}, "GetHome")

        if "data" not in data or data["data"] is None:
            # If we got a 400 error with "Required parameter is missing", raise to trigger retry
            if data and "errors" in data:
                error_msgs = [e.get("message", "") for e in data.get("errors", [])]
                if any("Required parameter is missing" in msg for msg in error_msgs):
                    raise Exception(f"Transient API error: {error_msgs}")
            return {}

        properties = data["data"]
        return {key.replace('home_', ''): properties[key] for key in properties if properties[key]}
