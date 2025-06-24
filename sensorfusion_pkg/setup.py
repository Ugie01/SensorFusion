from setuptools import find_packages, setup

package_name = 'sensorfusion_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ikhw',
    maintainer_email='ikhwanulabiyu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensorfusion1 = sensorfusion_pkg.sensorfusion1:main',
            'sensorfusion2 = sensorfusion_pkg.sensorfusion2:main'
        ],
    },
)
