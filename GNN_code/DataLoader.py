#batch to model

loader = DataLoader(dataset, batch_size=512, shuffle=True)

for batch in loader:
    batch