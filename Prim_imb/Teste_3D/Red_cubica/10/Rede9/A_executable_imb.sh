#!/bin/bash
echo "Start"
date
python 2_export_invasion_imbibition.py > output_imb.txt
date
python 3_calculating_krel_imb.py
date
echo "Done"

