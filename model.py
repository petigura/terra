from elixir import *

metadata.bind = "sqlite:///movies.sqlite"
metadata.bind.echo = True

class Mystars(Entity):
    name = Field(Text)
    oid  = Field(Integer)

    vsini = Field(Float(precision=5))
    teff = Field(Float(precision=5))

    o_abund = Field(Float(precision=5))

    c_abund = Field(Float(precision=5))

    o_staterrlo = Field(Float(precision=5))
    o_staterrhi = Field(Float(precision=5))

    c_staterrlo = Field(Float(precision=5))
    c_staterrhi = Field(Float(precision=5))

    o_nierrlo = Field(Float(precision=5))
    o_nierrhi = Field(Float(precision=5))
