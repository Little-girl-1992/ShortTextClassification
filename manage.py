from app import create_app
from flask_script import Manager


app = create_app()
app.debug = True

manager = Manager(app)


if __name__ == '__main__':
	manager.run()
