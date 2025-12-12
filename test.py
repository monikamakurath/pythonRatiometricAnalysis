import tifffile as tiff

file_path = "/Users/makurathm/My Drive (monikamakurath@gmail.com)/manuscripts/2025-ERiGlucoSnFR2/data/U2OS/0 glucose to glucose/20241122-U2OS-ERiGlucoSnFR-GluToGlu-05-Airyscan Processing-15-GluToGlu.tif"  # Replace with your actual file path
image = tiff.imread(file_path)

print("TIFF file shape:", image.shape)
