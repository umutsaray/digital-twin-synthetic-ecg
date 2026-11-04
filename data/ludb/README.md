# Local LUDB data

Raw LUDB 1.0.1 clinical recordings and annotations are intentionally not
redistributed in this repository. Download the database from PhysioNet and
extract it so that the local layout contains:

```text
data/ludb/lobachevsky-university-electrocardiography-database-1.0.1/
  ludb.csv
  data/1.hea
  data/1.dat
  data/1.ii
  ...
```

Then run:

```bash
python scripts/compare_pqrst_timing.py
```

The script reads only the Lead-II signal channel and `.ii` manual delineation
annotations. Aggregate derived results and the comparison figure are committed.
