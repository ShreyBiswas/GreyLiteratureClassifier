from functools import reduce

def split(text, max_len, overlap):

    new = [text[i : i + max_len] for i in range(0, len(text), max_len - overlap)]

    # last chunk might not start at the right point if the previous chunk got cut off
    if len(new) > 1:
        new[-1] = text[-max_len + overlap+1 :]

    return new

def recombine(chunks, overlap):
    return reduce(lambda acc, chunk: acc[:-overlap] + chunk, chunks)

def chunk_datapoint(xFeature,yLabel,max_len,overlap):
    text = xFeature["text"]

    chunks = split(text, max_len, overlap)

    newXFeatures = []
    newYLabels = []

    for chunk in range(len(chunks)):
        new = xFeature.copy()
        new["text"] = chunks[chunk]
        new['chunk_id'] = 0
        new['chunk_pos'] = chunk
        new['chunk_len'] = len(chunks)
        newXFeatures.append(new)
        newYLabels.append(yLabel)

    return newXFeatures, newYLabels

def chunk_dataset(xFeatures,yLabels,max_len,overlap):
    newXFeatures = []
    newYLabels = []

    for i in range(len(xFeatures)):
        new_xFeatures, new_yLabels = chunk_datapoint(xFeatures[i],yLabels[i],max_len,overlap)
        newXFeatures.extend(new_xFeatures)
        newYLabels.extend(new_yLabels)

    return newXFeatures, newYLabels



def recombine_chunks(chunks, overlap):
    # chunks may be out of order
    chunks = sorted(chunks, key=lambda x: (['chunk_pos']))
    text = recombine([x['text'] for x in chunks], overlap)
    new = chunks[0].copy()
    dict.pop(new, 'chunk_id')
    dict.pop(new, 'chunk_pos')
    dict.pop(new, 'chunk_len')
    new['text'] = text
    return new

def recombine_dataset(xFeatures,yLabels,overlap):
    # yLabels is a list of labels, with each label corresponding to a chunk
    chunk_groups = {}

    for i in range(len(xFeatures)):
        chunk_id = xFeatures[i]['chunk_id']
        if chunk_id not in chunk_groups:
            chunk_groups[chunk_id] = []
        chunk_groups[chunk_id].append(xFeatures[i])

    newXFeatures = []
    newYLabels = []

    y_pointer = 0
    for chunk_id in chunk_groups:
        new = recombine_chunks(chunk_groups[chunk_id], overlap)
        newXFeatures.append(new)
        newYLabels.append(yLabels[y_pointer])
        y_pointer += len(chunk_groups[chunk_id])

    return newXFeatures, newYLabels


if __name__ == '__main__':
    print(split("one two thref", 5, 2))

    xFeatures = [{"text": "hello world"}, {"text": "one two three"}]
    yLabels = [0, 1]

    max_len = 5
    overlap = 2

    newXFeatures, newYLabels = chunk_dataset(xFeatures, yLabels, max_len, overlap)
    print(newXFeatures)
    print(newYLabels)

    newXFeatures, newYLabels = recombine_xFeatures(newXFeatures, newYLabels, overlap)
    print(newXFeatures)
    print(newYLabels)
