# Pin CKAN's signing secrets from the environment (sourced by the stock
# start scripts from /docker-entrypoint.d, just before the server starts).
#
# Those scripts write fresh random secrets into the container's ckan.ini
# whenever it is created, so recreating the container — a deploy, a config
# change — silently signed out every session and invalidated every API token,
# including the one the mirror uses. With CKAN_SECRET_KEY set, the secrets
# survive. Unset, the stock random-per-container behaviour is kept.
if [ -n "${CKAN_SECRET_KEY:-}" ]; then
    ckan config-tool "$CKAN_INI" \
        "SECRET_KEY=${CKAN_SECRET_KEY}" \
        "WTF_CSRF_SECRET_KEY=${CKAN_SECRET_KEY}" \
        "api_token.jwt.encode.secret=string:${CKAN_SECRET_KEY}" \
        "api_token.jwt.decode.secret=string:${CKAN_SECRET_KEY}"
    echo "pin-secrets: signing secrets taken from CKAN_SECRET_KEY"
fi
