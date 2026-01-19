# TODO (riversnap)

Last updated: 2026-01-19

## Now (high priority)
- [ ] Candidate generation: support `k=None` (all within radius) safely
- [ ] Add `schema` support everywhere (`public.table` parsing)
- [ ] Ensure GiST indexes are created automatically on hydro + points tables

## Next (medium priority)
- [ ] Add diagnostics columns (pct, factor, pp, err, sim) consistently across backends
- [ ] Improve handling of duplicate columns in PostGIS queries
- [ ] Add CRS safety checks + explicit documentation of “metres vs degrees”

## Later (nice to have)
- [ ] Add support for additional hydrographies (HydroRIVERS, MERIT, etc.)
- [ ] Add text similarity distance component (river name matching)
- [ ] Example notebooks, e.g. 
  - [ ] Getting started
  - [ ] Benchmark notebook (GeoPandas vs PostGIS candidate generation)
- [ ] CLI: `riversnap postgis load-grit -c config.yml` 

## Bugs / Known issues

## Documentation
- [ ] Add PostGIS quickstart docs + docker-compose example
- [ ] Add “backend selection” guidance (filesystem vs PostGIS)
- [ ] Add API docs 

## Refactoring ideas

## Tests
- [ ] Unit test distance archetypes

## Release checklist
- [ ] Update version
- [ ] Update changelog
- [ ] Tag + release on GitHub
