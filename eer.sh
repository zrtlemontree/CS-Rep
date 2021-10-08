#!/bin/bash

paste -d  ' ' trials2 cos_score.score.original | awk -F ' ' '{print$6,$3}' | compute-eer  -
paste -d  ' ' trials2 cos_score.score.cs-rep  | awk -F ' ' '{print$6,$3}' | compute-eer  -
