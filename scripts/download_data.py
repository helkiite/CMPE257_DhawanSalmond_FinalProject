from datasets import load_dataset

amc = load_dataset("math-ai/amc23", split = "test") #validation dataset from CP Mobius
math500 = load_dataset("HuggingFaceH4/MATH-500", split = "test") #MATH 500 dataset

#printing AMC data details
print("AMC rows: ", len(amc))
print("AMC columns: ", amc.column_names)
print("AMC first example: ")
print(amc[0])

#printing MATH500 data details
print("\nMATH500 rows: ", len(math500))
print("MATH500 columns: ", math500.column_names)
print("MATH500 first example: ")
print(math500[0])