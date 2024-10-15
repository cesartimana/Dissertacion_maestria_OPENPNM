import pickle


with open('imbibition_process_24_07_18.pkl', 'rb') as fp:
    imbibition_info = pickle.load(fp)
    print(imbibition_info.keys())
    print(imbibition_info['status 1'].keys())
