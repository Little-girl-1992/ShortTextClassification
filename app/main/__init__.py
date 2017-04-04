from flask import Blueprint


main = Blueprint('main', __name__)

from . import view, main_code, exe_code
