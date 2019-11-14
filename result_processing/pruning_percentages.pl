#!/usr/bin/env perl
use strict;
use warnings;
use v5.24.1;

my $soln_count = 0;
my $blank = 0;
while (<>) {
    if (m#spec:specs/swinv_like/l(\d)/(.*).json#) {
        print "$2,$1";
        $blank = 1;
    }
    elsif (m#spec:.*/([^/]*).json#) {
        print"$1,0";
        $blank = 1;
    }

    if (/soln:.*/) {
        $soln_count += 1;
    }

    if (/stats:.*tested\((\d+)\), found\((\d+)\), failed\((\d+)\), pruned\((\d+)\), continued\((\d+)\)/) {
        my $tested = $1;
        my $found = $2;
        my $failed = $3;
        my $pruned = $4;
        my $continued = $5;
        if ($continued == 0 || $tested == 0) {
            if ($blank == 1) {
                print "\n";
                $blank = 0;
            }
            $soln_count = 0;
        }
        else {
            unless ($pruned == 0 && $continued == 1) {
                my $stat = 1.0 - ($continued * 1.0)/$tested;
                print ",$stat";
            }
        }
    }
}
