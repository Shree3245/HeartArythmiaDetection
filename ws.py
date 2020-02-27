##
#Start of dense Neural Network#
##


model=Sequential()
model.add(Dense(32,activation='relu',input_dim=X_train.shape[1]))
model.add(Dropout(rate=0.25))
model.add(Dense(1,activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)

model.fit(X_train,y_train,batch_size=32,epochs=5,verbose=1)

y_train_preds_dense=model.predict_proba(X_train,verbose=1)
y_valid_preds_dense=model.predict_proba(X_valid,verbose=1)

##
#End of DNN#
##

#start getting summary of DNN


print("Train")
print_report(y_train,y_train_preds_dense,thresh)
print("Valid")
print_report(y_valid,y_valid_preds_dense,thresh)
#End summary of DNN



###
# Start Transfer Learning #
###
from keras.layers import Bidirectional, LSTM
modelC = Sequential()
modelC.add(Conv1D(filters = 128, kernel_size = 5, activation = 'relu', input_shape=(160,1)))
modelC.add(Dropout(rate= 0.25))
modelC.add(Flatten())
modelC.add(Dense(1,activation='sigmoid'))

modelT = Sequential()
modelT.add(TimeDistributed(modelC))
modelT.add((LSTM(64, input_shape=(X_train_cnn.shape[1],X_train_cnn.shape[2]))))
modelT.add(Dropout(rate=0.25))
modelT.add(Dense(1,activation='sigmoid'))
modelT.compile(
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

modelT.fit(X_train_cnn[:10000],y_train[:10000],batch_size=32, epochs = 1, verbose =1)

y_train_preds_transfer = modelT.predict_proba(X_train_cnn[:10000],verbose =1)
y_valid_preds_transfer = modelT.predict_proba(X_valid_cnn,verbose =1)

print("Train")
print_report(y_train[:10000],y_train_preds_transfer,thresh)
print("Valid")
print_report(y_valid,y_valid_preds_transfer,thresh)
