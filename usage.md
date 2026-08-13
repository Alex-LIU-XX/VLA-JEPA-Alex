```
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA
conda activate VLA_JEPA

export DatasetsRoot=/share/home/tm866052366100000/a926312360/LXX/project/Datasets/lerobot/miku112
export DatasetsName=pick_food_pot_0725_1_offset_state

python scripts/convert_v3_to_v2_1.py \
  --input ${DatasetsRoot}/${DatasetsName} \
  --output ${DatasetsRoot}/${DatasetsName}_v2_1_full \
  --fps 10 \
  --chunk-size 100

```