#!/bin/bash
sudo docker run -it -v $(pwd):/home/root/cs470 cs470 bash -c "cd /home/root/cs470 && exec bash"

