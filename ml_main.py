from tkinter import *
import pandas as pd
from tkinter.filedialog import askopenfilename

def raise_frame(frame):
    frame.tkraise()

root = Tk()
root.minsize(width=500, height=500)
root.config(padx=10, pady=10)
root.title("Machine Learning Software")


f1 = Frame(root)
f2 = Frame(root)
f3 = Frame(root)
f4 = Frame(root)
Create_at = Frame(root)
Create_at_sub = Frame(root)

for frame in (f1, f2, f3, f4, Create_at, Create_at_sub):
    frame.grid(row=0, column=0, sticky='news')


Label(f1, text='Welcome to ML Softwase').pack()

Label(f2, text='FRAME 2').pack()
load_Button = Button(f2, text='Load data', command=lambda: f_3(f3))
load_Button.pack()
Label(Create_at, text='Create Attributes with CSV file').pack()
load_Button = Button(Create_at, text='Load data', command=lambda: Create_att(Create_at_sub))
load_Button.pack()

def f_3(f3):
    f3.destroy()
    f3 = Frame(root)
    f3.grid(row=0, column=0, sticky='news')
    Label(f3, text='FRAME 3').pack()
    name = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xlsx'))])
    if name:
        if name.endswith('.csv'):
            df = pd.read_csv(name)
            text = Text(f3)
            text.insert(INSERT, df)
            text.pack()
    buttin = Button(f3, text='Train Model', command=lambda: f_4(df, f4))
    buttin.pack()
    raise_frame(f3)

def Create_att(Create_at_sub):
    Create_at_sub.destroy()
    Create_at_sub = Frame(root)
    Create_at_sub.grid(row=0, column=0, sticky='news')
    Label(Create_at_sub, text='Create Attrubutes').grid(row=0, column=0, pady = 2, columnspan=2, sticky = N)
    
    name = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xlsx'))])
    if name:
        if name.endswith('.csv'):
            df = pd.read_csv(name)
            text = Text(Create_at_sub)
            text.insert(INSERT, df)
            text.grid(row=1, column = 0, columnspan=2, sticky = N, pady = 2)
    Label(Create_at_sub, text="Rows").grid(row=2, column = 0, sticky = E, pady = 2)
    Label(Create_at_sub, text="Columns").grid(row=3, column = 0, sticky = E, pady = 2)
    
    rows = Entry(Create_at_sub)
    columns = Entry(Create_at_sub)
    rows.grid(row=2, column=1, sticky = W, pady = 2)
    columns.grid(row=3, column=1, sticky = W, pady = 2)
    buttin = Button(Create_at_sub, text='Create Attribute', command=lambda: created_att(df, rows, columns))
    buttin.grid(row=4, column = 0, columnspan=2, sticky = N, pady = 2)
    raise_frame(Create_at_sub)
def created_att(df, rows, columns):
#    Create_at_sub.destroy()
#    Create_at_sub = Frame(root)
#    Create_at_sub.grid(row=0, column=0, sticky='news')
    rows = rows.get()
    columns = columns.get()
    print(rows, columns)
    

    
    

Label(f4, text='Model Results').pack()

def f_4(df, f4):
    dataset = df.iloc[:,:-1]
    labels = df.iloc[:,-1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, 
                                                random_state=42)
    # Here print the shapee of dataset. Mean how much sample in the dataset and how much features in the data
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    from sklearn import svm
    model = svm.SVC(gamma='auto', probability=True, verbose=True)
    model = model.fit(X_train, y_train)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    result = model.score(X_test, y_test)
    print("Prediction the labels of the unseen datset.....")
    output = model.predict(X_test)
#    print(output)
#    probability = model.predict_proba(X_test)
#    print("confidence how % is False and True" )
#    print("False 0,  True 1 \n", probability)
#    print("Accuracy of the Model =", result)
#    print("Model classification results \n", classification_report(output, y_test))
#    print("Confusion matrix \n", confusion_matrix(output, y_test))
    text = Text(f4)
    text.insert(INSERT, 'Prediction the labels of the unseen datset.....\n')
    text.insert(INSERT, output)
    text.insert(INSERT, '\n'+classification_report(output, y_test))
    text.insert(INSERT, confusion_matrix(output, y_test))
    text.pack()
    raise_frame(f4)
    

raise_frame(f1)
test = Menu(root)
test.add_command(label='Create Atributes', command=lambda:raise_frame(Create_at))
test.add_command(label='Import Dataset', command=lambda:raise_frame(f2))
test.add_command(label='Model Output', command=lambda:raise_frame(f3))
test.add_command(label='Visulize')
test.add_command(label='Export Dataset')
root.config(menu=test)
root.mainloop()