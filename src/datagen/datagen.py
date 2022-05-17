from datagen.synthetic.single import sphere, swissroll
from datagen.synthetic.multiple import intertwinedswissrolls, concentricspheres, wellseparatedspheres

dtype = {
    "single-sphere": sphere.RandomSphere,
    "single-swissroll": swissroll.RandomSwissRoll,
    "ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "inf-ittw-swissrolls": intertwinedswissrolls.IntertwinedSwissRolls,
    "conc-spheres": concentricspheres.ConcentricSpheres,
    "ws-spheres": wellseparatedspheres.WellSeparatedSpheres,
    "inf-ws-spheres": wellseparatedspheres.WellSeparatedSpheres,
    "inf-conc-spheres": concentricspheres.ConcentricSpheres
}


