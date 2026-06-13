const SITE_TITLE = "shall we swim?";

export const LOCATIONS_PAGE_TITLE = `Open water swimming locations | ${SITE_TITLE}`;
export const LOCATION_NOT_FOUND_TITLE = `Location not found | ${SITE_TITLE}`;
export const PAGE_NOT_FOUND_TITLE = `Page not found | ${SITE_TITLE}`;

export function locationPageTitle(swimLocation: string) {
  return `${swimLocation} swim conditions | ${SITE_TITLE}`;
}
