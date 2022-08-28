import os
import io
from jupyterthemes import jtplot
from .logger import logger
from typing import Literal
import rpy2.robjects as robjects

current_jupytertheme_style: Literal['dark', 'light'] = 'light'

def displayImage(src, background=None, **kwargs):
    from IPython.display import Image, SVG, display
    from IPython.core.display import HTML
    if src[-4:].lower() == '.svg':
        htmlstr = '<div class="jp-RenderedSVG jp-OutputArea-output " data-mime-type="image/svg+xml">'
        svgstr = SVG(data=src)._repr_svg_()
        if not isinstance(background, type(None)):
            bgstr = f'style="background-color:{background}"'
            svgstr = '<svg ' + bgstr + svgstr[4:]
        htmlstr += svgstr + '</div>'
        display(HTML(htmlstr))
    else:
        img = Image(data=src)
        fmt = img.format
        if fmt == 'png':
            htmlstr = '<div class="jp-RenderedImage jp-OutputArea-output ">'
            htmlstr += f'\n<img src="data:image/png;base64,{img._repr_png_()}"></div>'
            if not isinstance(background, type(None)):
                htmlstr = htmlstr[:-7] + f'style="background-color:{background}"' + htmlstr[-7:]
            display(HTML(htmlstr))
        else:
            logger.warning('background for none PNG image will be ignored')
            display(img)


def setDarkModePlotting(forceDark=False, forceWhite=False):
    if (current_jupytertheme_style == 'dark' and not forceDark) or forceWhite:
        logger.warning('Change plot style to default')
        jtplot.reset()
        current_jupytertheme_style = 'light'
    else:
        logger.warning('Change plot style to dark (monokai)')
        jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
        current_jupytertheme_style = 'dark'


def writeRSessionInfo(fileName, overwrite=True):
    def printToString(*args, **kwargs):
        with io.StringIO() as output:
            print(*args, file=output, **kwargs)
            contents = output.getvalue()
        return contents

    try:
        sinfo = 'Attached packages in current R session:\n' + '=' * 80 + '\n'
        for l in robjects.r('sessionInfo()["otherPkgs"]')[0]:
            x = printToString(l).split('\n')
            y = [a for a in x if len(a) > 0 and not a.startswith('-- File:')]
            z = '\n'.join(y) + '\n\n' + "=" * 80 + '\n'
            sinfo += z
    except TypeError:
        logger.info('No R packages loaded in current session')
        return 0
    logger.info(sinfo)
    if not overwrite:
        if os.path.isfile(fileName):
            logger.info(f'File {fileName} exists, will not overwrite.')
            return 0
    with open(fileName, 'w') as fh:
        fh.write(sinfo)
    return 1

