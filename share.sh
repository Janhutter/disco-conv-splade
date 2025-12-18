#!/bin/bash


setfacl -R -m u:scur1719:rwX /scratch-shared/disco/
setfacl -R -d -m u:scur1719:rwX /scratch-shared/disco/

setfacl -R -m u:scur1691:rwX /scratch-shared/disco/
setfacl -R -d -m u:scur1691:rwX /scratch-shared/disco/

setfacl -R -m u:scur1710:rwX /scratch-shared/disco/
setfacl -R -d -m u:scur1710:rwX /scratch-shared/disco/