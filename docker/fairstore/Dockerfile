FROM ckan/ckan-dev:2.11.3

# Keep the extension revision deterministic and aligned with the CKAN 2.11
# version used by the hosted Fair Store. This revision's CI covers CKAN 2.11.
ARG CKANEXT_HIERARCHY_SHA=53c1ee74805d79aef7c9c01cc513eb881cb8928f

USER root
RUN pip install --no-cache-dir \
    "git+https://github.com/ckan/ckanext-hierarchy.git@${CKANEXT_HIERARCHY_SHA}#egg=ckanext-hierarchy"

USER ckan
