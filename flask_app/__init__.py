from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.route('/test')
    def test():
        return "it's alive!"

    from . import art_select_view
    app.register_blueprint(art_select_view.bp)

    return app
