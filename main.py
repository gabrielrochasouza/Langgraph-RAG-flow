from dotenv import load_dotenv

from graph.graph import app

load_dotenv()

if __name__ == '__main__':
    print('App is working')
    print(app.invoke(input={"question": "What is agent memory?"}))
