#!/usr/bin/env perl
use strict;
use warnings;
use v5.24.1;

my $spec = "NOSPEC";
my $level = 0;
print "Spec,Spec level,Basis name,Tested states,Failed merge check,Pruned,Rejected,Continued,Solutions\n";
while (<>) {
    if (m#spec:specs/swinv_like/l(\d)/(.*).json#) {
        $spec = $2;
        $level = $1;
    }
    elsif (m#spec:.*/([^/]*).json#) {
        $spec = $1;
        $level = 0;
    }
    if (/stats:.*\((.*)\) tested\((\d+)\), found\((\d+)\), failed\((\d+)\), pruned\((\d+)\), continued\((\d+)\)/) {
        my $name = $1;
        my $tested = $2;
        my $found = $3;
        my $failed = $4;
        my $pruned = $5;
        my $continued = $6;
        my $rejected = $failed + $pruned;
        print "$spec,$level,$name,$tested,$failed,$pruned,$rejected,$continued,";
        if ($found != 0) {
            print "$found\n";
        }
        else {
            print "\n";
        }
    }
}
