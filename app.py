from flask import Flask , request,redirect, url_for , render_template
import pandas as pd
import pymysql

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction',methods =['GET','POST'])
def prediction():
    if request.method == 'POST':
        msg=''
        
        Student_Age = request.form['Student_Age'] # default empty
        Sex = int(request.form['Sex'])
        High_School_Type = int(request.form['High_School_Type'])
        Scholarship = int(request.form['Scholarship'])
        Additional_Work = int(request.form['Additional_Work'])
        Sports_activity =  int(request.form['Sports_activity'])
        Transportation =int( request.form['Transportation'])
        Weekly_Study_Hours = int(request.form['Weekly_Study_Hours'])
        Attendance = int( request.form['Attendance'])
        Reading = int(request.form['Reading'])
        Notes = int(request.form['Notes'])
        Listening_in_class = int(request.form['Listening_in_class'])
        Project_work = int(request.form['Project_work'])
        test_data=[Student_Age, Sex , High_School_Type , Scholarship , Additional_Work,Sports_activity,Transportation, Weekly_Study_Hours , Attendance,Reading ,Notes,Listening_in_class, Project_work]
        print(test_data)
        from sklearn.model_selection import train_test_split 
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression 
        from sklearn import tree 
        from sklearn.metrics import accuracy_score 
        data = pd.read_csv("static/Students.csv")
        data = data.drop('Student_ID', axis = 1)
        data.head()
        data.tail()
        data.info()
        data.describe()
        data['Sex'].replace(['Female', 'Male'],[0,1], inplace=True)
        data['Additional_Work'].replace(['Yes', 'No'],[1,0],inplace=True)
        data['Sports_activity'].replace(['Yes', 'No'], [1,0], inplace = True)
        data['Transportation'].replace(['Public', 'Private', 'Bus'],[0,1,2], inplace = True)
        data['Reading'].replace(['Yes', 'No'], [1,0], inplace=True)
        data['Attendance'].replace(['Never','Sometimes','Always'],[0,1,2], inplace=True) 
        data['Notes'].replace(['Yes','No'],[1,0],inplace=True)
        data['Listening_in_Class'].replace(['Yes','No'],[1,0],inplace= True)      
        data['Attendance'] = pd.to_numeric(data['Attendance'], errors = 'coerce')
        data['Notes'] = pd.to_numeric(data['Notes'], errors='coerce')
        data['Listening_in_Class'] = pd.to_numeric(data['Listening_in_Class'], errors = 'coerce')
        
        #Missing values la change akum 1 or 0 ha 
        
        data['Attendance'].fillna(1, inplace=True)
        data['Notes'].fillna(0, inplace=True)
        data['Listening_in_Class'].fillna(0, inplace=True)
        
        data['Project_work'].replace(['Yes', 'No'],[1,0], inplace=True)
        data['High_School_Type'].replace(['State', 'Private','Other'], [1,0,2], inplace = True)
        data['Student_Age'].replace({
            '18':18,
            '19-22':20,
            '23-27':25
        }, inplace=True)

        #data['Scholarship'] = data['Scholarship'].replace('None','0%')
        #data['Scholarship'] = data['Scholarship'].str.replace('%','').astype(int)
        
        
        
        #ithula first uh Grade ha category nu oru function use panni grade la numerric ha change panna vaichu irukom 
        data['Grade'] = data['Grade'].astype('category').cat.codes
        
        
        
        # ithula Scholarship percentage la iruthurahta change pannurom 
        #first empty la irukuratha 0% la mathum 
        #second la athoda type enna nu solluthu
        #third la athoda replacement nadakuu
        #finally athoda replacement ku apro type ha check pannurathuku marupadium type nu kettu int nu solluraga
        data['Scholarship'] = data['Scholarship'].fillna('0%')
        data['Scholarship'] = data['Scholarship'].astype(str)
        data['Scholarship'] = data['Scholarship'].str.replace('%','')
        data['Scholarship'] = data['Scholarship'].astype(int)
        
        
        x= data.drop(["Grade"], axis =1)
        y= data["Grade"]
        
        data.head()
        
        x.shape
        y.shape
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train.shape
        x_test.shape
        
        
        #model = KNeighborsClassifier(n_neighbors=5)
        #using logistic regression algorthim for predicting the accuracy of the output and max-iter na enna na 1000 time iteration pannna 
        model = LogisticRegression(max_iter=1000)
        #model =tree.DecisionTreeClassifier()      
        
        
        x_test.shape
        
        print(data.dtypes)
        
        model.fit(x_train, y_train)
        
        predict_output = model.predict(x_test)
        print(predict_output)
        print(y_test)
        
        accuracy = accuracy_score(y_test, predict_output)
        accuracy
        
        predictions = model.predict([test_data])
        
        print(predictions)
        
        if predictions[0] <= 1:
            output="Good performance"
        elif predictions[0] == 2:
            output="Average performance"
        else:
            output="Poor performance"
        return render_template('result.html' , output = output )
    return render_template('prediction.html')

            
if __name__ == '__main__':
    app.run(port=5000, debug=True)

         