# -----------------------------------------------------------------------------
#  flow/orfs_extract.tcl  --  the ONLY parser of an ORFS routed result
# -----------------------------------------------------------------------------
#  Run inside the local OpenROAD (parameters via env, OpenROAD eats CLI argv):
#     ORFS_DESIGN=gcd ORFS_ODB=<6_final.odb> ORFS_SDC=<6_final.sdc> \
#     ORFS_LIB=<nangate45.lib> ORFS_OUTDIR=<dir> ORFS_VDD=1.1 ORFS_VSS=0.0 \
#     openroad -exit -no_init flow/orfs_extract.tcl
#
#  Emits (JSON/CSV, consumed by prism/orfs.py):
#     <outdir>/<design>_geom.json      die, insts, macros, straps, pads
#     <outdir>/<design>_paths.json     per-endpoint slack + path instances
#     <outdir>/<design>_vdd_nodes.csv  PDNSim per-node (x_um, y_um, voltage)
#
#  Uses OpenDB (odb) + OpenSTA + PSM. No hand-rolled DEF parsing.
# -----------------------------------------------------------------------------

proc env_or {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
  return $default
}

set design  [env_or ORFS_DESIGN design]
set odbf    [env_or ORFS_ODB    ""]
set sdcf    [env_or ORFS_SDC    ""]
set libf    [env_or ORFS_LIB    ""]
set outdir  [env_or ORFS_OUTDIR "."]
set vdd     [env_or ORFS_VDD    1.1]
set vss     [env_or ORFS_VSS    0.0]
file mkdir $outdir

puts "orfs_extract: design=$design odb=$odbf"
if {$libf ne "" && [file exists $libf]} { read_liberty $libf }
read_db $odbf
if {$sdcf ne "" && [file exists $sdcf]} { catch { read_sdc $sdcf } }

set block [ord::get_db_block]
set dbu   [$block getDefUnits]
proc u {v} { global dbu; return [expr {double($v)/$dbu}] }

# ---------------- geometry -------------------------------------------------
set da [$block getDieArea]
set die [list [u [$da xMin]] [u [$da yMin]] [u [$da xMax]] [u [$da yMax]]]

set insts {}
set macros {}
foreach inst [$block getInsts] {
  set nm [$inst getName]
  set mst [$inst getMaster]
  set mn [$mst getName]
  set bb [$inst getBBox]
  set x [u [$bb xMin]]; set y [u [$bb yMin]]
  set w [expr {[u [$bb xMax]] - $x}]; set h [expr {[u [$bb yMax]] - $y}]
  set isblock [expr {[$mst isBlock] ? 1 : 0}]
  lappend insts [list n $nm m $mn x $x y $y w $w h $h b $isblock]
  if {$isblock} {
    lappend macros [list n $nm x0 $x y0 $y x1 [expr {$x+$w}] y1 [expr {$y+$h}]]
  }
}

# power/ground special-wire rectangles -> "straps"
set straps {}
foreach net [$block getNets] {
  set st [$net getSigType]
  if {$st ne "POWER" && $st ne "GROUND"} { continue }
  foreach swire [$net getSWires] {
    foreach wseg [$swire getWires] {
      if {[$wseg isVia]} { continue }
      lappend straps [list x0 [u [$wseg xMin]] y0 [u [$wseg yMin]] \
                           x1 [u [$wseg xMax]] y1 [u [$wseg yMax]] \
                           net $st]
    }
  }
}

# bump / pad pins from block terminals that carry a placement
set pads {}
foreach bt [$block getBTerms] {
  if {[$bt getSigType] ne "POWER" && [$bt getSigType] ne "GROUND"} { continue }
  foreach bp [$bt getBPins] {
    foreach bx [$bp getBoxes] {
      lappend pads [list x [u [expr {([$bx xMin]+[$bx xMax])/2}]] \
                         y [u [expr {([$bx yMin]+[$bx yMax])/2}]] \
                         net [$bt getSigType]]
    }
  }
}

proc jesc {s} { return [string map [list \\ {\\} \" {\"} \t {\t} \n {\n} \r {}] $s] }
proc jkv {pairs} {
  set o {}
  foreach {k v} $pairs {
    if {[string is double -strict $v]} { lappend o "\"$k\":$v" } \
    else { lappend o "\"$k\":\"[jesc $v]\"" }
  }
  return "{[join $o ,]}"
}
proc jlist {rows} {
  set o {}
  foreach r $rows { lappend o [jkv $r] }
  return "\[[join $o ,]\]"
}

set fh [open "$outdir/${design}_geom.json" w]
puts $fh "{"
puts $fh "\"die_um\": \[[join $die ,]\],"
puts $fh "\"dbu\": $dbu,"
puts $fh "\"insts\": [jlist $insts],"
puts $fh "\"macros\": [jlist $macros],"
puts $fh "\"straps\": [jlist $straps],"
puts $fh "\"pads\": [jlist $pads]"
puts $fh "}"
close $fh
puts "orfs_extract: geom -> ${design}_geom.json  (insts=[llength $insts] macros=[llength $macros] straps=[llength $straps] pads=[llength $pads])"

# ---------------- timing paths ------------------------------------------
# plain-text report to stdout (no `redirect` command in this build); the
# python side parses the block between the markers out of OpenROAD's stdout.
puts ">>>PRISM_PATHS_BEGIN<<<"
catch {
  report_checks -path_delay max -slack_max 1e30 \
    -group_path_count 400 -endpoint_path_count 1 \
    -fields {input_pins} -format full_clock_expanded
} emsg
if {$emsg ne ""} { puts "PRISM_PATHS_ERR: $emsg" }
puts ">>>PRISM_PATHS_END<<<"

# ---------------- PDNSim per-node voltage ------------------------------
set irok 0
foreach opt {
  {analyze_power_grid -net VDD -voltage_file OUT}
  {analyze_power_grid -net VDD -output_voltage_file OUT}
  {analyze_power_grid -net VDD -error_file OUT}
} {
  set cmd [string map [list OUT "$outdir/${design}_vdd_nodes.csv"] $opt]
  if {![catch { eval $cmd } emsg]} {
    if {[file exists "$outdir/${design}_vdd_nodes.csv"]} { set irok 1; break }
  } else {
    puts "orfs_extract: ($cmd) -> $emsg"
  }
}
if {!$irok} {
  puts "orfs_extract: WARN could not dump per-node voltage; prism/orfs.py will fall back"
} else {
  puts "orfs_extract: irmap -> ${design}_vdd_nodes.csv"
}
exit 0
