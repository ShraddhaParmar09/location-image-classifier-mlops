# 3. Dataset research
Find and compare suitable datasets:

# Dataset size
# Number of classes/locations
# Geographic coverage
# Labels
# License
# Train/validation/test split
# Class imbalance
# Availability

> Three open-source datasets are best suited for this project. Since you are doing research comparing a Custom CNN against Transfer Learning, your best option is to download a controlled subset of one of these databases rather than the entire multi-gigabyte repository.

# -> d:\location-image-classifier-mlops\docs\image.png

----------------------------------------------------------------------------------------------------------------

> My current decision says:

MIT Places365 → 5-class balanced subset

with classes like:

kitchen, playground, beach, mountain, street

That makes the project a scene classifier, not really a location classifier.

If we keep the project as Location Image Classifier

Change Part 3 so that your candidate datasets contain actual geographic/location labels.

For example, depending on the exact V1 you choose:

Image
 ↓
Geographic/location dataset
 ↓
Location label
 ↓
Model
 ↓
Predicted location

Your dataset comparison table can remain — you just need to research location-relevant datasets instead of using Places365 as the final dataset.

You can still keep Places365 in your research as a reference/baseline dataset, but I wouldn't make it your final dataset if the project remains a location classifier.