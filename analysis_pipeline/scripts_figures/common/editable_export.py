"""Editable Matplotlib vector exports with Cairo as the default rendering backend.

Matplotlib's Cairo SVG backend converts text into paths. Route SVG output
through Matplotlib's SVG backend; route PDF output through its PDF backend,
embedding TrueType fonts. PNG remains rendered by the configured Cairo backend.
"""
from pathlib import Path
import matplotlib


def save_figure(fig, filename, **kwargs):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower().lstrip('.')
    with matplotlib.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42,
                                'ps.fonttype': 42}):
        if ext in ('pdf', 'svg'):
            fig.savefig(path, backend=ext, **kwargs)
        else:
            fig.savefig(path, **kwargs)
    return path
