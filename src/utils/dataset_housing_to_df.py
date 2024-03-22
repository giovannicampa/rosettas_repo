import glob
from glob import glob

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset():
    images = glob("datasets/Houses-dataset/Houses Dataset/*.jpg", recursive=True)
    images.sort()

    outputs = ["nr_bedrooms", "nr_bathrooms", "area", "price", "else"]
    output_features = [out for out in outputs if not out == "else"]
    df = pd.read_csv(
        "datasets/Houses-dataset/Houses Dataset/HousesInfo.txt",
        header=None,
        delimiter=" ",
        names=outputs,
    )

    df["image_path"] = None
    df.drop(["else"], inplace=True, axis=1)

    image_types = ["bathroom", "bedroom", "frontal", "kitchen"]

    for image_type in image_types:
        df[f"image_path_{image_type}"] = df.index.map(
            lambda x: f"datasets/Houses-dataset/Houses Dataset/{x+1}_{image_type}.jpg"
        )

    scaler = StandardScaler()
    df[output_features] = scaler.fit_transform(df[output_features])

    return df
