from pathlib import Path
import pandas as pd
import shutil

from . import path_exists, get_files_from
from .image import IMG_EXTENSIONS

HTML_WIDTH = 2000


def generate_html_grid(files, n_col=1, max_width=300):
    # width = min(HTML_WIDTH / n_col, max_width)
    # height = width
    html = "<center>\n<table>\n<tr>\n"
    for k, f in enumerate(files):
        ext = Path(f).suffix
        f = str(f)
        html += f'<td><a href="{f}" target="_blank">'
        if ext in ['.mp4', '.ogg', '.webm']:
            html += f'<video autoplay loop muted controls playsinline src="{f}" width=100%/>'
        else:
            html += f'<img src="{f}" alt="{f}" width=100%/>'
        html += f'</a></td>\n'
        if k % n_col == n_col - 1:
            html += '</tr>\n<tr>\n'
    html += '</tr>\n</table>\n<center>'
    return html


class HtmlImagesPageGenerator:
    def __init__(self, input_dir, output_name, title=None, cluster_prefix=None, with_captions=False, nb_col=10):
        self.input_dir = path_exists(input_dir).absolute()
        self.output_name = output_name
        self.title = title or 'Extraction results'
        self.cluster_prefix = cluster_prefix or 'Cluster'
        self.with_captions = with_captions
        self.nb_col = nb_col

    def run(self):
        file = self._init_file_and_write_html_header()
        directories = sorted(set(map(lambda p: p.parent, get_files_from(
            self.input_dir, IMG_EXTENSIONS, recursive=True))))
        metric_file = self.input_dir / 'metrics.tsv'
        if metric_file.exists():
            df = pd.read_csv(metric_file, sep='\t', index_col=False)
        else:
            df = None
        for d in directories:
            file.write('<table>\n')
            subtitle = '{} {}'.format(self.cluster_prefix, d.name)
            if df is not None:
                name = d.name if self.input_dir.name != d.name else '.'
                subtitle += ' - mIoU = {:.4f}'.format(df[df['dir_name'] == name]['iou_class_1'].values[0])
            file.write('\t\t<h2>{}</h2>\n'.format(subtitle))
            file.write('\t<tr>\n')
            img_files = get_files_from(d, valid_extensions=IMG_EXTENSIONS, sort=True)
            for j, img in enumerate(map(lambda f: f.relative_to(self.input_dir), img_files)):
                caption = '<br /> {}'.format(img.stem.replace('_', ' ')) if self.with_captions else ''
                file.write('\t\t<td> <a href=\"{}\" target="_blank" title="{}"> <img src=\"{}\" alt="{}" /></a>{}\n'
                           .format(img, img.name, img, img.name, caption))
                if j % self.nb_col == self.nb_col - 1:
                    file.write('\t</tr>\n')
                    file.write('\t<tr>\n')
            file.write('\t</tr>\n')
            file.write('</table>\n')

        file.write('</center>\n</div>\n </body>\n</html>\n')
        file.close()

        shutil.copy(Path(__file__).parent / 'style.css', self.input_dir)

    def _init_file_and_write_html_header(self):
        file = open(self.input_dir / self.output_name, 'w')
        file.write('<html>\n')
        file.write('<head>\n')
        file.write('\t<title></title>\n')
        file.write('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
        file.write('\t<meta name=\"robots\" content=\"index, follow\" />\n')
        file.write('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
        file.write('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
        file.write('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
        file.write('\t<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" media=\"screen\" />\n')
        file.write('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
        file.write('</head>\n')
        file.write('<body>\n')
        file.write('<div id="website">\n')
        file.write('<center>\n')
        file.write('\t<div class=\"blank\"></div>\n')
        file.write('\t<h1>\n')
        file.write('\t\t{}\n'.format(self.title))
        file.write('\t</h1>\n')
        file.write('</center>\n')
        file.write('<div class=\"blank\"></div>\n')
        file.write('<center>\n')
        file.write('<div>\n')
        file.write('</div>\n')

        return file
