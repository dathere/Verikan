# Mark CKAN's session and remember-me cookies Secure when the site is served
# over HTTPS (sourced by the stock start scripts from /docker-entrypoint.d).
#
# The image's ckan.ini ships both as false, and uppercase Flask keys cannot be
# set through the envvars plugin (it lowercases names), hence config-tool.
# Local dev on http://localhost keeps them false: a browser never sends a
# Secure cookie back over plain HTTP, so nobody could stay logged in.
#
# The image also ships REMEMBER_COOKIE_SAMESITE = None, which lets another site
# make a remembered sysadmin's browser send the one-year cookie with a
# cross-site POST (CKAN's DataStore dictionary form is CSRF-exempt); Lax
# works over plain HTTP too, so it is set everywhere.
ckan config-tool "$CKAN_INI" "REMEMBER_COOKIE_SAMESITE=Lax"
case "${CKAN_SITE_URL:-}" in
    https://*)
        ckan config-tool "$CKAN_INI" \
            "SESSION_COOKIE_SECURE=true" \
            "REMEMBER_COOKIE_SECURE=true"
        echo "secure-cookies: session cookies marked Secure"
        ;;
esac
