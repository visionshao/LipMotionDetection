import os
import dlib

current_path = os.getcwd()

faces_path = current_path + '/examples/faces'

options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

training_xml_path = os.path.join(faces_path, "training_with_face_landmarks.xml")
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)
print("Training accuracy:{0}".format(dlib.test_shape_predictor(training_xml_path, "predictor.dat")))

testing_xml_path = os.path.join(faces_path,"testing_with_face_landmarks.xml")
print("Testing accuracy:{0}".format(dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))