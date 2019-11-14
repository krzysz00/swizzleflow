#!/usr/bin/env perl
use strict;
use warnings;
use v5.24.1;

my $spec = "NOSPEC";
my $level = 0;
my $mat_time = 0.0;
print "Spec,Spec level,Matrix time,Search time,Total time\n";
while (<>) {
    if (m#spec:specs/swinv_like/l(\d)/(.*).json#) {
        $spec = $2;
        $level = $1;
    }
    elsif (m#spec:.*/([^/]*).json#) {
        $spec = $1;
        $level = 0;
    }
    if (/(mul|build|load):.* \[(.*)\]/) {
        $mat_time += $2;
    }
    if (/search:.* \[(.*)\]/) {
        my $search_time = $1;
        my $total = $mat_time + $search_time;
        print "$spec,$level,$mat_time,$search_time,$total\n";
        $mat_time = 0.0;
    }
}
