# Raytracing in One Weekend (in CUDA)

After completing the "Raytracing in One Weekend" tutorial by Peter Shirley, I decided to implement the code in CUDA to learn more about parallel programming and GPU acceleration.

The original tutorial can be found [here](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

Obligatory render of the final scene (1000 samples per pixel, 500 bounce cap, 2K resolution):
![Render of the finale scene](./finalerender.png?raw=true)

### Notes
* I only had a few days to learn CUDA and implement this, so the project structure (and code) isn't perfect but the code works.
* I spent some time with Nsight Compute to look into performance bottlenecks (mostly high register allocation for my code) and to optimize the code.
* I also spent some time familiarizing myself with the CUDA memory model and how to optimize memory access patterns and access speed using shared and constant memory for instance.

All in all, I learned a _lot_ and had fun doing this project. I hadn't really considered GPU programming before so I hadn't realized that there were many differences between it and "normal" CPU programming. It's definitely a different way of thinking in some aspects.
I look forward to using this knowledge in future projects!

