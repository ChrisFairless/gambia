# Collection and processing of data for CLIMADA

The scripts in these folders gather and format the data for the following:

## Hazard:

### Flood: Aqueduct
Data from the World Resources Institute. Gridded data at different return periods, downloaded and formatted for a Hazard object

### TODO:

- **JRC**: I've heard the EU's JRC flood data (historic only) is better for much of Africa. I'd like to compare
- **Historic events**: for comparison and validation. Either from satellite footprints, or from event perimeters on e.g. Humanitarian Data Exchange


## Exposure

### Population: Global Human Settlement Layer
Downloaded from the JRC data portal for the relevant patches of the globe.

The data are available at multiple resolutions and projections. I've chosen 3 arcsec (very high resolution) and the WGS84 projection, since CLIMADA uses this as default so we don't have to reproject.



