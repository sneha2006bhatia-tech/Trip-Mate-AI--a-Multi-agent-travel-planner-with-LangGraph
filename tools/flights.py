import os
import re
import certifi
import airportsdata
import requests
import pycountry
from dotenv import load_dotenv
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
DEFAULT_ORIGIN_IATA=os.getenv("DEFAULT_ORIGIN_IATA", "DEL")
BASE_URL="https://api.aviationstack.com/v1/flights"
AIRPORTS = airportsdata.load('IATA')
from datetime import datetime, timezone

MAX_API_CALLS = 6  # cap API calls per search (free plans have small quotas)

# Countries with several airports; all of them are searched
COUNTRY_AIRPORTS = {
    "JP": ["NRT", "HND", "KIX"],
    "BD": ["DAC", "CGP"],
    "IN": ["DEL", "BOM", "BLR"],
    "US": ["JFK", "LAX", "ORD"],
    "GB": ["LHR", "LGW"],
    "AE": ["DXB", "AUH"],
    "CN": ["PEK", "PVG", "CAN"],
    "TH": ["BKK", "DMK"],
    "KR": ["ICN", "GMP"],
}

# City preferred airport
CITY_MAIN_AIRPORT = {
    "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR",
    "bengaluru": "BLR", "chennai": "MAA", "hyderabad": "HYD", "kolkata": "CCU",
    "dhaka": "DAC", "tokyo": "NRT", "london": "LHR", "paris": "CDG",
    "dubai": "DXB", "singapore": "SIN", "new york": "JFK", "sydney": "SYD",
    "bangkok": "BKK", "kathmandu": "KTM",
}
COUNTRY_ALIASES = {
    "usa": "US",
    "u.s.a": "US",
    "u.s": "US",
    "america": "US",
    "united states": "US",

    "uk": "GB",
    "u.k.": "GB",
    "britain": "GB",
    "england": "GB",

    "uae": "AE",
    "dubai": "AE",

    "south korea": "KR",
    "korea": "KR",

    "russia": "RU",
    "vietnam": "VN",
    "bangladesh": "BD",
    "india": "IN",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "nepal": "NP",
    "qatar": "QA",
    "saudi arabia": "SA",
    "turkey": "TR",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES"
}
COUNTRY_MAIN_AIRPORTS = {
    "US": "JFK",
    "GB": "LHR",
    "AE": "DXB",
    "KR": "ICN",
    "RU": "SVO",
    "VN": "SGN",
    "BD": "DAC",
    "IN": "DEL",
    "JP": "NRT",
    "CN": "PEK",
    "SG": "SIN",
    "MY": "KUL",
    "TH": "BKK",
    "ID": "CGK",
    "NP": "KTM",
    "QA": "DOH",
    "SA": "JED",
    "TR": "IST",
    "CA": "YYZ",
    "AU": "SYD",
    "DE": "FRA",
    "FR": "CDG",
    "IT": "FCO",
    "ES": "MAD"
}
def clean_text(text:str)->str:
    text=text.lower().strip()
    text=re.sub(r"[^a-z0-9\s]", "", text)
    text=re.sub(r"\s+", " ", text)
    stop_words=[
        "flight","flights","ticket","tickets","trip","travel","plan","complete","day","days","including","hotels","hotel","sightseeing","under","budget","info","information"
    ]
    words=[w for w in text.split() if w not in stop_words]
    return " ".join(words).strip()
def country_name_to_code(text: str):
    text = clean_text(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2
    except LookupError:
        pass

    # Detect country name inside longer text
    for country in pycountry.countries:
        country_name = country.name.lower()
        if country_name in text:
            return country.alpha_2

    for alias, code in COUNTRY_ALIASES.items():
        if alias in text:
            return code

    return None
def airport_country_matches(airport: dict, country_code: str) -> bool:
    airport_country = str(airport.get("country", "")).upper().strip()

    if airport_country == country_code:
        return True

    try:
        country = pycountry.countries.get(alpha_2=country_code)

        if country and airport_country.lower() == country.name.lower():
            return True

    except Exception:
        pass

    return False
def get_best_airport_for_country(country_code: str):
    preferred = COUNTRY_MAIN_AIRPORTS.get(country_code)

    if preferred and preferred in AIRPORTS:
        return preferred

    candidates = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0

            if "international" in name:
                score += 50

            if "intl" in name:
                score += 40

            if "capital" in name:
                score += 20

            if city:
                score += 5

            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]
def get_best_airport_for_city(location: str):
    location_clean = clean_text(location)

    if not location_clean:
        return None

    city_matches = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        name = str(airport.get("name", "")).lower()
        city = str(airport.get("city", "")).lower()

        score = 0

        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70

        if location_clean in name:
            score += 50

        if "international" in name:
            score += 10

        if score > 0:
            city_matches.append((score, iata))

    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]

    return None

def resolve_location_to_iata(location: str):
    """
    Converts country/city/airport/IATA into an IATA code.
 
    Examples:
        Bangladesh -> DAC
        Japan      -> NRT
        Dhaka      -> DAC
        Tokyo      -> NRT
        DAC        -> DAC
    """
    if not location:
        return None
 
    raw_location = location.strip()
 
    # Direct IATA code
    # (skip words like "usa" that are also country aliases)
    if re.fullmatch(r"[A-Za-z]{3}", raw_location) and raw_location.lower() not in COUNTRY_ALIASES:
        code = raw_location.upper()
        if code in AIRPORTS:
            return code
 
    location_clean = clean_text(raw_location)
    if not location_clean:
        return None
 
    # City preferred airport
    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]
 
    # Country preferred airport
    country_code = country_name_to_code(location_clean)
    if country_code:
        airport = get_best_airport_for_country(country_code)
        if airport:
            return airport
 
    # Exact city match from airport database
    city_matches = []
    for iata, airport in AIRPORTS.items():
        if not iata:
            continue
 
        city = str(airport.get("city", "")).lower().strip()
        name = str(airport.get("name", "")).lower().strip()
 
        score = 0
        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70
 
        if location_clean in name:
            score += 50
 
        if "international" in name:
            score += 10
 
        if score > 0:
            city_matches.append((score, iata))
 
    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]
 
    return None
 
 
def resolve_location_to_airports(location: str):
    """Like resolve_location_to_iata, but returns several airports for big countries."""
    raw = (location or "").strip()
    if re.fullmatch(r"[A-Za-z]{3}", raw) and raw.lower() not in COUNTRY_ALIASES and raw.upper() in AIRPORTS:
        return [raw.upper()]
 
    clean = clean_text(raw)
    if clean in CITY_MAIN_AIRPORT:
        return [CITY_MAIN_AIRPORT[clean]]
 
    country_code = country_name_to_code(clean)
    if country_code and country_code in COUNTRY_AIRPORTS:
        return COUNTRY_AIRPORTS[country_code]
 
    code = resolve_location_to_iata(raw)
    return [code] if code else []
 
 
def get_origin_iata(location=None):
    if location:
        result = resolve_location_to_iata(location)
        if not result:
            raise ValueError(f"Could not identify origin airport: {location}")
        return result
    return DEFAULT_ORIGIN_IATA
 
 
# --------------------------------------------------------------------------
# Route parsing
# --------------------------------------------------------------------------
def _find_mentions_with_pos(query: str):
    """Returns [(start, end, name)] sorted by position, longest names win overlaps."""
    low = query.lower()
    names = set(CITY_MAIN_AIRPORT) | set(COUNTRY_ALIASES)
    names |= {c.name.lower() for c in pycountry.countries}
 
    found, used = [], []
 
    def overlaps(s, e):
        return any(s < ue and e > us for us, ue in used)
 
    # Longest first so "new delhi" beats "delhi" and "south korea" beats "korea"
    for name in sorted(names, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(name)}\b", low):
            if not overlaps(m.start(), m.end()):
                used.append((m.start(), m.end()))
                found.append((m.start(), m.end(), name))
 
    # Explicit IATA codes written in capitals, e.g. "DAC to NRT"
    for m in re.finditer(r"\b[A-Z]{3}\b", query):
        if m.group() in AIRPORTS and not overlaps(m.start(), m.end()):
            used.append((m.start(), m.end()))
            found.append((m.start(), m.end(), m.group()))
 
    return sorted(found)
 
 
def find_location_mentions(query: str):
    """City / country / IATA names found in the query, in order, without duplicates."""
    unique_mentions = []
    for _, _, name in _find_mentions_with_pos(query):
        if name not in unique_mentions:
            unique_mentions.append(name)
    return unique_mentions
 
 
def parse_route_text(query: str):
    """Returns (dep_text, arr_text) as place names (either can be None)."""
    low = query.lower()
    mentions = []
    for start, _, name in _find_mentions_with_pos(query):
        if name not in [m[1] for m in mentions]:
            mentions.append((start, name))
 
    if not mentions:
        return None, None
 
    dep = arr = None
    unassigned = []
    for start, name in mentions:
        prefix = low[:start].rstrip()
        if prefix.endswith("from") and dep is None:
            dep = name
        elif prefix.endswith("to") and arr is None:
            arr = name
        else:
            unassigned.append(name)
 
    for name in unassigned:
        if arr is None and dep is not None:
            arr = name
        elif dep is None and arr is not None:
            dep = name
        elif dep is None and arr is None:
            # "DAC NRT" -> first is departure, second is arrival; a lone place is the destination
            if len(unassigned) >= 2 and name == unassigned[0]:
                dep = name
            else:
                arr = name
    return dep, arr
 
 
def parse_route(query: str):
    """
    Returns:
        dep_iata, arr_iata
 
    Can return:
        None, None  -> global live flights
        DAC, NRT    -> filtered route
        DAC, None   -> all flights from DAC
        None, NRT   -> all flights to NRT
    """
    dep_text, arr_text = parse_route_text(query)
    dep_iata = resolve_location_to_iata(dep_text) if dep_text else None
    arr_iata = resolve_location_to_iata(arr_text) if arr_text else None
    return dep_iata, arr_iata
 
 
# --------------------------------------------------------------------------
# Flights
# --------------------------------------------------------------------------
def validate_flight_date(flight_date):
    if not flight_date:
        return None
    try:
        datetime.strptime(flight_date, "%Y-%m-%d")
        return flight_date
    except ValueError:
        raise ValueError("Invalid date. Please use YYYY-MM-DD format.")
 
 
def get_scheduled_departure(flight):
    scheduled = (flight.get("departure") or {}).get("scheduled")
    if not scheduled:
        return None
    try:
        t = datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None
 
 
def format_flight(flight: dict):
    airline = (flight.get("airline") or {}).get("name") or "Unknown airline"
    flight_number = (flight.get("flight") or {}).get("iata") or "Unknown flight number"
    status = flight.get("flight_status") or "Unknown"
 
    dep = flight.get("departure") or {}
    arr = flight.get("arrival") or {}
 
    dep_airport = dep.get("airport") or "Unknown departure airport"
    dep_iata = dep.get("iata") or "Unknown"
    dep_terminal = dep.get("terminal") or "N/A"
    dep_gate = dep.get("gate") or "N/A"
    dep_scheduled = dep.get("scheduled") or "Unknown"
    dep_delay = dep.get("delay")
    dep_delay_text = f"{dep_delay} minutes" if dep_delay is not None else "N/A"
 
    arr_airport = arr.get("airport") or "Unknown arrival airport"
    arr_iata = arr.get("iata") or "Unknown"
    arr_terminal = arr.get("terminal") or "N/A"
    arr_gate = arr.get("gate") or "N/A"
    arr_scheduled = arr.get("scheduled") or "Unknown"
    arr_delay = arr.get("delay")
    arr_delay_text = f"{arr_delay} minutes" if arr_delay is not None else "N/A"
 
    return f"""Airline: {airline}
Flight: {flight_number}
Status: {status}
 
Departure:
- Airport: {dep_airport}
- IATA: {dep_iata}
- Terminal: {dep_terminal}
- Gate: {dep_gate}
- Scheduled: {dep_scheduled}
- Delay: {dep_delay_text}
 
Arrival:
- Airport: {arr_airport}
- IATA: {arr_iata}
- Terminal: {arr_terminal}
- Gate: {arr_gate}
- Scheduled: {arr_scheduled}
- Delay: {arr_delay_text}"""
 
 
def fetch_flights(dep_iata, arr_iata, flight_date=None):
    """One API call for one airport pair. Returns the raw list of flights."""
    if not API_KEY:
        raise RuntimeError("AVIATIONSTACK_API_KEY is missing.")
 
    params = {"access_key": API_KEY, "limit": 100}
    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata
    if flight_date:
        params["flight_date"] = flight_date  # paid plans only
 
    try:
        response = requests.get(BASE_URL, params=params, timeout=30, verify=certifi.where())
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Flight API request failed: {e}")
 
    if data.get("error"):
        raise RuntimeError(f"AviationStack API error: {data['error']}")
 
    flights = data.get("data") or []
    print(f"  {dep_iata or '*'} -> {arr_iata or '*'}: API returned {len(flights)} records")
    return flights
 
 
def search_flights(destination: str, origin: str = None, flight_date: str = None,
                   limit: int = 10):
    """Search upcoming scheduled flights between origin and destination."""
 
    # Step 1: figure out the route
    if origin is None:
        dep_text, arr_text = parse_route_text(destination)
        if not arr_text and not dep_text:
            return "Please provide a valid route, e.g. 'India to USA' or 'DAC to NRT'."
        origin = dep_text or DEFAULT_ORIGIN_IATA
        destination = arr_text or destination
 
    # Step 2: resolve airports (countries expand to several airports)
    origins = resolve_location_to_airports(origin)
    destinations = resolve_location_to_airports(destination)
 
    if not origins:
        return f"Could not identify origin airport: {origin}"
    if not destinations:
        return f"Could not identify destination airport: {destination}"
 
    pairs = [(o, d) for o in origins for d in destinations if o != d]
    if not pairs:
        return "Origin and destination cannot be the same."
 
    # Step 3: validate date
    try:
        date_string = validate_flight_date(flight_date)
    except ValueError as error:
        return str(error)
 
    print(f"\nSearching {len(pairs[:MAX_API_CALLS])} route(s): "
          + ", ".join(f"{o}->{d}" for o, d in pairs[:MAX_API_CALLS]))
 
    # Step 4: fetch
    all_flights, errors = [], []
    for dep_iata, arr_iata in pairs[:MAX_API_CALLS]:
        try:
            all_flights += fetch_flights(dep_iata, arr_iata, date_string)
        except RuntimeError as error:
            errors.append(str(error))
 
    if errors and not all_flights:
        return "Unable to retrieve flights:\n" + "\n".join(errors)
 
    route_label = f"{'/'.join(origins)} to {'/'.join(destinations)}"
 
    if not all_flights:
        return (f"The API returned no flights for {route_label}. "
                "There may be no direct flights on this route in AviationStack's data "
                "(many country pairs are only reachable with a connection). "
                "Try a specific airport pair such as DEL to BOM to confirm your key works.")
 
    # Step 5: clean up (drop codeshare duplicates and wrong routes)
    valid_dep = {o for o, _ in pairs}
    valid_arr = {d for _, d in pairs}
    seen, flights = set(), []
    for f in all_flights:
        if f.get("flight", {}).get("codeshared"):
            continue
        dep = (f.get("departure") or {}).get("iata") or ""
        arr = (f.get("arrival") or {}).get("iata") or ""
        if dep.upper() not in valid_dep or arr.upper() not in valid_arr:
            continue
        key = ((f.get("flight") or {}).get("iata"), (f.get("departure") or {}).get("scheduled"))
        if key in seen:
            continue
        seen.add(key)
        flights.append(f)
 
        # Step 6: keep only scheduled flights (not landed, not in the air)
    scheduled = [f for f in flights if f.get("flight_status") == "scheduled"]

    # Step 7: soonest departure first, take the top N
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    scheduled.sort(key=lambda f: get_scheduled_departure(f) or far_future)
    top = scheduled[:limit]

    if not top:
        return f"No scheduled flights found for {route_label}."

    # Step 8: print in the same style as your screenshots
    return "\n\n---\n\n".join(format_flight(f) for f in top)
if __name__ == "__main__":
    print("Flights: Bangladesh to Japan")
    print("=" * 80)
    print(search_flights(destination="Japan", origin="Bangladesh", limit=10))
 
    # Natural-language usage also works:
    # print(search_flights("trip to Japan from Bangladesh"))
    # print(search_flights("DAC to NRT"))
 