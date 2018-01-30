import platform
import sys

from ase.utils import import_module, FileNotFoundError
from ase.utils import search_current_git_hash
from ase.io.formats import filetype, all_formats, UnknownFileTypeError
from ase.io.ulm import print_ulm_info
from ase.io.bundletrajectory import print_bundletrajectory_info


class CLICommand:
    short_description = 'Print information about files or system'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('filenames', nargs='*')
        parser.add_argument('-v', '--verbose', action='store_true')

    @staticmethod
    def run(args):
        if not args.filenames:
            print_info()
            return

        n = max(len(filename) for filename in args.filenames) + 2
        for filename in args.filenames:
            try:
                format = filetype(filename)
            except FileNotFoundError:
                format = '?'
                description = 'No such file'
            except UnknownFileTypeError:
                format = '?'
                description = '?'
            else:
                description, code = all_formats.get(format, ('?', '?'))

            print('{:{}}{} ({})'.format(filename + ':', n,
                                        description, format))
            if args.verbose:
                if format == 'traj':
                    print_ulm_info(filename)
                elif format == 'bundletrajectory':
                    print_bundletrajectory_info(filename)


def print_info():
    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]
    for name in ['ase', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            versions.append((name, 'no'))
        else:
            # Search for git hash
            githash = search_current_git_hash(module)
            if githash is None:
                githash = ''
            else:
                githash = '-{:.10}'.format(githash)
            versions.append((name + '-' + module.__version__ + githash,
                            module.__file__.rsplit('/', 1)[0] + '/'))

    for a, b in versions:
        print('{:25}{}'.format(a, b))
