#!/bin/bash
echo "Start"
date
python 1_export_invasion_drainage.py
date
python 2_export_invasion_imbibition.py
date
python 3_calculating_krel_drn.py
date
python 3_calculating_krel_imb.py
date
echo "Done"

