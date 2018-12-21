#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
https://github.com/AotY/DSTC7-End-to-End-Conversation-Modeling/blob/simpler/data_extraction/src/Makefile.official.targets
"""

targets_dict = {
    # train
    "OFFICIAL_TRAIN_CONVOS": "data-official/2011-01.convos.txt data-official/2011-02.convos.txt data-official/2011-03.convos.txt data-official/2011-04.convos.txt data-official/2011-05.convos.txt data-official/2011-06.convos.txt data-official/2011-07.convos.txt data-official/2011-08.convos.txt data-official/2011-09.convos.txt data-official/2011-10.convos.txt data-official/2011-11.convos.txt data-official/2011-12.convos.txt data-official/2012-01.convos.txt data-official/2012-02.convos.txt data-official/2012-03.convos.txt data-official/2012-04.convos.txt data-official/2012-05.convos.txt data-official/2012-06.convos.txt data-official/2012-07.convos.txt data-official/2012-08.convos.txt data-official/2012-09.convos.txt data-official/2012-10.convos.txt data-official/2012-11.convos.txt data-official/2012-12.convos.txt data-official/2013-01.convos.txt data-official/2013-02.convos.txt data-official/2013-03.convos.txt data-official/2013-04.convos.txt data-official/2013-05.convos.txt data-official/2013-06.convos.txt data-official/2013-07.convos.txt data-official/2013-08.convos.txt data-official/2013-09.convos.txt data-official/2013-10.convos.txt data-official/2013-11.convos.txt data-official/2013-12.convos.txt data-official/2014-01.convos.txt data-official/2014-02.convos.txt data-official/2014-03.convos.txt data-official/2014-04.convos.txt data-official/2014-05.convos.txt data-official/2014-06.convos.txt data-official/2014-07.convos.txt data-official/2014-08.convos.txt data-official/2014-09.convos.txt data-official/2014-10.convos.txt data-official/2014-11.convos.txt data-official/2014-12.convos.txt data-official/2015-01.convos.txt data-official/2015-02.convos.txt data-official/2015-03.convos.txt data-official/2015-04.convos.txt data-official/2015-05.convos.txt data-official/2015-06.convos.txt data-official/2015-07.convos.txt data-official/2015-08.convos.txt data-official/2015-09.convos.txt data-official/2015-10.convos.txt data-official/2015-11.convos.txt data-official/2015-12.convos.txt data-official/2016-01.convos.txt data-official/2016-02.convos.txt data-official/2016-03.convos.txt data-official/2016-04.convos.txt data-official/2016-05.convos.txt data-official/2016-06.convos.txt data-official/2016-07.convos.txt data-official/2016-08.convos.txt data-official/2016-09.convos.txt data-official/2016-10.convos.txt data-official/2016-11.convos.txt data-official/2016-12.convos.txt",

    "OFFICIAL_TRAIN_FACTS": "data-official/2011-01.facts.txt data-official/2011-02.facts.txt data-official/2011-03.facts.txt data-official/2011-04.facts.txt data-official/2011-05.facts.txt data-official/2011-06.facts.txt data-official/2011-07.facts.txt data-official/2011-08.facts.txt data-official/2011-09.facts.txt data-official/2011-10.facts.txt data-official/2011-11.facts.txt data-official/2011-12.facts.txt data-official/2012-01.facts.txt data-official/2012-02.facts.txt data-official/2012-03.facts.txt data-official/2012-04.facts.txt data-official/2012-05.facts.txt data-official/2012-06.facts.txt data-official/2012-07.facts.txt data-official/2012-08.facts.txt data-official/2012-09.facts.txt data-official/2012-10.facts.txt data-official/2012-11.facts.txt data-official/2012-12.facts.txt data-official/2013-01.facts.txt data-official/2013-02.facts.txt data-official/2013-03.facts.txt data-official/2013-04.facts.txt data-official/2013-05.facts.txt data-official/2013-06.facts.txt data-official/2013-07.facts.txt data-official/2013-08.facts.txt data-official/2013-09.facts.txt data-official/2013-10.facts.txt data-official/2013-11.facts.txt data-official/2013-12.facts.txt data-official/2014-01.facts.txt data-official/2014-02.facts.txt data-official/2014-03.facts.txt data-official/2014-04.facts.txt data-official/2014-05.facts.txt data-official/2014-06.facts.txt data-official/2014-07.facts.txt data-official/2014-08.facts.txt data-official/2014-09.facts.txt data-official/2014-10.facts.txt data-official/2014-11.facts.txt data-official/2014-12.facts.txt data-official/2015-01.facts.txt data-official/2015-02.facts.txt data-official/2015-03.facts.txt data-official/2015-04.facts.txt data-official/2015-05.facts.txt data-official/2015-06.facts.txt data-official/2015-07.facts.txt data-official/2015-08.facts.txt data-official/2015-09.facts.txt data-official/2015-10.facts.txt data-official/2015-11.facts.txt data-official/2015-12.facts.txt data-official/2016-01.facts.txt data-official/2016-02.facts.txt data-official/2016-03.facts.txt data-official/2016-04.facts.txt data-official/2016-05.facts.txt data-official/2016-06.facts.txt data-official/2016-07.facts.txt data-official/2016-08.facts.txt data-official/2016-09.facts.txt data-official/2016-10.facts.txt data-official/2016-11.facts.txt data-official/2016-12.facts.txt",

    # dev
    "OFFICIAL_DEV_CONVOS": "data-official/2017-01.convos.txt data-official/2017-02.convos.txt data-official/2017-03.convos.txt",

    "OFFICIAL_DEV_FACTS": "data-official/2017-01.facts.txt data-official/2017-02.facts.txt data-official/2017-03.facts.txt",

    # Note: validation == dev set, but validation is a subset used for automatic evaluation
    # valid
    "OFFICIAL_VALID_CONVOS": "data-official-valid/2017-01.convos.txt data-official-valid/2017-02.convos.txt data-official-valid/2017-03.convos.txt",

    "OFFICIAL_VALID_FACTS": "data-official-valid/2017-01.facts.txt data-official-valid/2017-02.facts.txt data-official-valid/2017-03.facts.txt",

    # test refs
    "OFFICIAL_TEST_CONVOS": "data-official-test/2017-04.convos.txt data-official-test/2017-05.convos.txt data-official-test/2017-06.convos.txt data-official-test/2017-07.convos.txt data-official-test/2017-08.convos.txt data-official-test/2017-09.convos.txt data-official-test/2017-10.convos.txt data-official-test/2017-11.convos.txt",

    "OFFICIAL_TEST_FACTS": "data-official-test/2017-04.facts.txt data-official-test/2017-05.facts.txt data-official-test/2017-06.facts.txt data-official-test/2017-07.facts.txt data-official-test/2017-08.facts.txt data-official-test/2017-09.facts.txt data-official-test/2017-10.facts.txt data-official-test/2017-11.facts.txt",

    "OFFICIAL_TEST_REFS": "data-official-test/2017-04.refs.txt data-official-test/2017-05.refs.txt data-official-test/2017-06.refs.txt data-official-test/2017-07.refs.txt data-official-test/2017-08.refs.txt data-official-test/2017-09.refs.txt data-official-test/2017-10.refs.txt data-official-test/2017-11.refs.txt"
}
