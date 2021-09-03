class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9]) # age
    input_bounds.append([0, 7]) # workclass
    input_bounds.append([0, 39]) #69 for THEMIS  fnlwgt
    input_bounds.append([0, 15]) # education
    input_bounds.append([0, 6]) # marital_status
    input_bounds.append([0, 13]) # occupation
    input_bounds.append([0, 5]) # relationship
    input_bounds.append([0, 4]) # race
    input_bounds.append([0, 1]) #  sex
    input_bounds.append([0, 99]) # capital_gain
    input_bounds.append([0, 39]) # capital_loss
    input_bounds.append([0, 99]) # hours_per_week
    input_bounds.append([0, 39]) # native_country

    input_bounds_size=[]
    for x in input_bounds:
        input_bounds_size.append(x[1]-x[0])
    # the name of each feature
    feature_name = ["age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
                    "savings_status", "employment", "installment_commitment", "sex", "other_parties",
                     "residence", "property_magnitude", "age", "other_payment_plans", "housing",
                    "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                    "month", "duration", "campaign", "pdays", "previous", "poutcome"]

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class compas:
    """
    Configuration of dataset Bank Marketing
    """
    from aif360.datasets.compas_dataset import CompasDataset
    import numpy as np
    cd = CompasDataset()

    # the size of total features
    params = len(cd.features[0])

    # the valid religion of each feature
    input_bounds = []
    categorical_features = []
    for i in range(params):
        input_bounds.append([int(np.min(cd.features[:,i])),int(np.max(cd.features[:,i]))])
        categorical_features.append(i)

    # the name of each feature
    feature_name = cd.feature_names

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    # categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

class meps:
    """
    Configuration of dataset Bank Marketing
    """
    import numpy as np
    from aif360.datasets.meps_dataset_panel21_fy2016 import MEPSDataset21
    cd = MEPSDataset21()
    cd.features = np.delete(cd.features, [10], axis=1)  # axis=1 删除列，axis=0 删除行


    # the size of total features
    params = len(cd.features[0])

    # the valid religion of each feature
    input_bounds = []
    categorical_features = []
    for i in range(params):
        input_bounds.append([int(np.min(cd.features[:,i])),int(np.max(cd.features[:,i]))])
        categorical_features.append(i)

    # the name of each feature
    feature_name = cd.feature_names

    # the name of each class
    class_name = ["no", "yes"]

    # specify the categorical features with their indices
    # categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

