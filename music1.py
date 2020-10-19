import librosa as lb
import os
import json

DATASET  = "C:\\Genre Dataset"    
JSON = "C:\\mfcc_data.json" 

SAMPLE_RATE = 22050
DUR = 30    
SAMPLES = SAMPLE_RATE*DUR



def create_mfcc_data(dataset_path, json_filepath, n_mfcc=13, n_fft=2048, hop_length=512, segments=5):
    
    
    
    data = {
        "mapping" : [],
        "mfcc" : [],
        "label" : []
    }
    
    segment_size = SAMPLES//segments
    mfcc_vectors = math.ceil(segment_size/hop_length)
    
    
    
    for i,(dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            
            genre_path = dirpath.split('\\')
            genre = genre_path[-1]
            data["mapping"].append(genre)
            
            print("Processing {} ({}/10)-".format(genre, i))
            
            
            
            for f in filenames:
                
                filepath = os.path.join(dirpath, f)
                signal, sr = lb.load(filepath, sr=SAMPLE_RATE)    
                
                for s in range(segments):
                    
                    start = s*segment_size    
                    end = start + segment_size    
                    
                    mfcc = lb.feature.mfcc(signal[start:end], sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)   
                    mfcc = mfcc.T
                    
                    
                    if len(mfcc) == mfcc_vectors:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)
                    
    
                  
    with open(json_filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print("All Genres Processed...Done!!!")

    


    
