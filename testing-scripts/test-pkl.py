import pickle


with open("validcand.pkl", "rb") as f:
    resume_name_map = pickle.load(f)

print(resume_name_map)