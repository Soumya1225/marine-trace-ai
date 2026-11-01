# Design Decisions

## D1: Use Sentinel-1 SAR imagery
- **Reason:** SAR works day and night and through clouds; oil smooths the sea surface and appears as dark patches.

## D2: Use Krestenitis et al. (2019) dataset for training
- **Reason:** Pixel-level labels with 5 classes, including look-alikes, which helps reduce false alarms.

## D3: Use free AIS data from US / Danish waters
- **Reason:** AIS for Indian waters is not freely available; US (MarineCadastre) and Danish (DMA) data are free and well documented.

## D4: Late fusion design (image features + AIS features combined at attribution stage)
- **Reason:** The two data types have very different formats, so each is processed separately and joined at the end.
