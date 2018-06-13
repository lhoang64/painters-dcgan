from flask import Blueprint, render_template, request
import os
import json

bp = Blueprint('art_select', __name__, url_prefix='/art_select', static_folder='static')
styles = os.listdir(os.path.join(bp.static_folder, 'images'))
available_styles = [' '.join(s.split('_')[1::]) for s in styles]


@bp.route('/styles', methods=['GET'])
def art_select():
    return render_template('/styles_view.html', styles=available_styles)


@bp.route('/styles/<art_style>', methods=['GET'])
def art_style(art_style):
    if art_style in available_styles:
        images = os.listdir(os.path.join(bp.static_folder, 'images', 'processed_{}'.format('_'.join(art_style.split(' ')))))
        src_path = os.path.join(bp.static_url_path, 'images', 'processed_{}'.format('_'.join(art_style.split(' '))))
        return render_template('/art_select_view.html', path=json.dumps(src_path), style=art_style, images=json.dumps(images))
    else:
        return "Style not available in dataset."


@bp.route('/save', methods=['POST'])
def save():
    style = request.form['style']
    selected_images = {"selected_images": request.form['selected_images']}
    print(selected_images)
    output_dir = os.path.join(bp.static_folder, 'user_selected')
    filename = os.path.join(output_dir, '{0}_v{1}.json'.format(style, len(os.listdir(output_dir))))
    with open(filename, 'w') as f:
        json.dump(selected_images, f)
    return render_template('/styles_view.html', styles=available_styles)


