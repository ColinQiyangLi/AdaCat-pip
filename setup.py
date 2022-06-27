from setuptools import setup
 
setup(
    name = 'adacat',
    packages = ['adacat'],
    version = '0.2',
    description = 'Adaptive categorical distribution',
    author='Qiyang Li',
    author_email='colin.qiyang.li@gmail.com',
    url='https://github.com/ColinQiyangLi/AdaCat-pip',
    dependencies = ["torch"],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS'
    ]
)
