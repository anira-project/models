// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/build_info.h>
#include <anira/abi/status.h>
#include <anira/abi/version.h>
#include <gtest/gtest.h>

// The anira library this test links matches the headers it was compiled against.
TEST(Anira, AbiMatchesHeaders) {
    EXPECT_EQ(anira_check_abi(ANIRA_ABI_VERSION), ANIRA_OK);
}

TEST(Anira, IsTheThreeLine) {
    EXPECT_GE(ANIRA_VERSION_MAJOR, 3);
}
