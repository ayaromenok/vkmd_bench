reorganize file structure:
 - move C to src
 - move GLSL to src/shaders
 - move compiled GLSL/SPV to (build)/shaders subflder
 - move resulted CSV somewhere
 - separate ini file to settins.ini and banchmark.ini
 - merge 'matmul.comp' and 'elemop.comp'

add pause between tests for cool-down device
