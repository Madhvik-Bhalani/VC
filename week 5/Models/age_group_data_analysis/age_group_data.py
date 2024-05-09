import pandas as pd

dataset_df = pd.read_csv("Dataset/diabetes.csv")

bins = [20, 30, 40, 50, 60, float("inf")]  # last bin is for 60+

# Define labels for age groups
labels = ["20-30", "31-40", "41-50", "51-60", "61+"]

# Create 'age_group' column using pd.cut()
dataset_df["age_group"] = pd.cut(
    dataset_df["Age"], bins=bins, labels=labels, right=False
)

def calculate_avg_data(age):

    # Get the age group using pd.cut()
    age_group = pd.cut([age], bins=bins, labels=labels, right=False)[0]

    # Filter the DataFrame to include only rows in the specified age group
    age_group_data = dataset_df[dataset_df["age_group"] == age_group]

    # calculate avg data for each factors
    avg_age_group_data = []
    Pregnancies = 0
    Glucose = 0
    BloodPressure = 0
    SkinThickness = 0
    Insulin = 0
    BMI = 0
    DiabetesPedigreeFunction = 0
    count = 0
    for i in age_group_data.values:
        count = count + 1
        Pregnancies = Pregnancies + i[0]
        Glucose = Glucose + i[1]
        BloodPressure = BloodPressure + i[2]
        SkinThickness = SkinThickness + i[3]
        Insulin = Insulin + i[4]
        BMI = BMI + i[5]
        DiabetesPedigreeFunction = DiabetesPedigreeFunction + i[6]

    avg_age_group_data.append(["Pregnancies", round(Pregnancies / count)])
    avg_age_group_data.append(["Glucose", round(Glucose / count)])
    avg_age_group_data.append(["BloodPressure", round(BloodPressure / count)])
    avg_age_group_data.append(["SkinThickness", round(SkinThickness / count)])
    avg_age_group_data.append(["Insulin", round(Insulin / count)])
    avg_age_group_data.append(["BMI", round(BMI / count, 1)])
    avg_age_group_data.append(["DiabetesPedigreeFunction", round(DiabetesPedigreeFunction / count, 3)])
    avg_age_group_data.append(["Age_group", age_group_data.values[0][-1]])

    return avg_age_group_data
