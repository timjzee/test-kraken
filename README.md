# test-kraken

Very little documentation about training using the API, so I used the cli:

First create a binary training file
```bash
result=$(ls /vol/tensusers/timzee/kraken/trainingfiles/*.xml)
ketos compile --workers 10 -f page -o dataset_large.arrow $result
```

Then train the network using recommended parameter settings:
```bash
ketos train --precision 16 --augment --workers 4 -d cuda -f binary --min-epochs 20 -w 0 -s '[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 Mp2,2 Cr3,9,64 Do0.1,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do.1,2 Lbx200 Do]' -r 0.0001 dataset_large.arrow
```

This resulted in pretty good validation accuracy (around .93), but we can stop training earlier, so perhaps change to `--min-epochs 10`.