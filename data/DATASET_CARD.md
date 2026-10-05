# Dataset Card: PlantVillage for PlantGuard Lab

## Source and intended use

PlantVillage is an open-access collection introduced by Hughes and Salathé and
used by Mohanty, Hughes and Salathé (2016). The maintained source repository is
`https://github.com/spMohanty/PlantVillage-Dataset`. It reports 54,306 images,
14 crop species, 26 diseases, and 38 crop-condition classes.

This project uses the colour configuration for supervised image
classification and controlled robustness evaluation. It is suitable for
research baselines, not direct claims of field-level diagnostic accuracy.

## Files created locally

- `raw/PlantVillage-Dataset/raw/color/`: source images
- `processed/{train,val,test}/<class>/`: split images
- `processed/manifest.csv`: path, label, crop, disease, leaf group, split
- `processed/split_summary.csv`: image counts by class and split

## Split policy

Default proportions are 70% train, 15% validation, and 15% test with seed 42.
Where `leaf-map.json` provides a leaf identity, all images of that leaf remain
in one split. Otherwise the source file stem is used as a conservative group.

## Known limitations and bias

- Mostly controlled backgrounds and lighting
- Unequal class frequencies
- Limited geographic, seasonal, camera, and cultivar metadata
- Crop-condition labels are not a complete taxonomy of plant disease
- Image-level predictions do not establish disease severity or treatment

Metadata collected by the application must be used for monitoring and
stratified evaluation. It must not be used to manufacture post-hoc accuracy
claims or infer precise location.

## Ethics

The images contain plants rather than human participants. The application uses
coarse region labels only, avoids precise GPS collection, and stores no
personally identifiable information. Outputs include a research-use disclaimer.

## Citation

Mohanty, S. P., Hughes, D. P. and Salathé, M. (2016), "Using deep learning for
image-based plant disease detection", Frontiers in Plant Science, 7, 1419.

